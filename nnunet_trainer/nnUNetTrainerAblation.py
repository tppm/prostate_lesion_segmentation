from pathlib import Path
import numpy as np
import torch
import yaml
from hydra.utils import get_class, instantiate
from monai.transforms import Compose
from torch import nn
import segmentation
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from bicc_loss import VectorizedBiCCLoss, VectorizedCCLoss

CONF_DIR = Path(segmentation.__path__[0]) / "conf"


class LesionVsRest(nn.Module):
    """Apply a binary instance loss to lesion (the last label) vs every other class.

    loss: a binary, 2-channel `bicc_loss` module (blob loss, CC-DiceCE or BiCC) taking an int16
        label map; non-lesion logits are folded into background by logsumexp, so BackSplit classes count as
        background and a binary run passes through unchanged; computed on float32 logits
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, logits, target):
        logits = logits.float()
        lesion_label = logits.shape[1] - 1
        background_logits = torch.logsumexp(logits[:, :-1], dim=1, keepdim=True)
        binary_logits = torch.cat([background_logits, logits[:, -1:]], dim=1)
        lesion_target = (target == lesion_label).to(torch.int16)
        return self.loss(binary_logits, lesion_target)


class LesionLoss(nn.Module):
    """nnU-Net's CE and Dice plus a lesion-vs-rest instance term, each summed over nnU-Net's deep-supervision levels.

    dc_and_ce: nnU-Net's DC_and_CE_loss; its `ce` and `dc` are the first two terms
    instance_loss: `LesionVsRest`-wrapped binary instance loss, fed float32 logits
    ds_weights: nnU-Net's per-level weights; zero-weight levels are skipped
    """

    def __init__(self, dc_and_ce, instance_loss, ds_weights):
        super().__init__()
        self.ce = dc_and_ce.ce
        self.dc = dc_and_ce.dc
        self.instance_loss = instance_loss
        self.ds_weights = ds_weights

    def forward(self, outputs, targets):
        ce = 0.0
        dice = 0.0
        instance = 0.0
        for level_weight, logits, target in zip(self.ds_weights, outputs, targets):
            if level_weight == 0:
                continue
            ce = ce + level_weight * self.ce(logits, target[:, 0])
            dice = dice + level_weight * self.dc(logits, target, loss_mask=None)
            instance = instance + level_weight * self.instance_loss(logits.float(), target)
        return ce.float() + dice.float() + instance.float()


class nnUNetTrainerAblation(nnUNetTrainer):
    """nnUNetTrainer set up by `trainer_config`, one `segmentation/conf/nnunet_trainer/<name>.yaml` (registered below
    as `nnUNetTrainer_<name>`); every key left out keeps nnU-Net's behaviour.

    instance_loss: `_target_` binary instance loss added to nnU-Net's CE + Dice through `LesionVsRest` (`LesionLoss`);
        CC and BiCC also get the plans' spacing and patch size
    rand_filter: `segmentation/conf/rand_filter/<name>.yaml`, applied on the GPU after nnU-Net's augmentation with the
        slice axis (the plans' coarsest spacing) last
    """

    trainer_config = {}

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.rand_filter = None
        if "rand_filter" in self.trainer_config:
            steps_path = CONF_DIR / "rand_filter" / f"{self.trainer_config['rand_filter']}.yaml"
            steps = []
            for step in yaml.safe_load(steps_path.read_text())["steps"]:
                steps.append(instantiate(step))
            self.rand_filter = Compose(steps)
            self.slice_axis = 2 + int(np.argmax(self.configuration_manager.spacing))

    def _build_loss(self):
        nnunet_loss = super()._build_loss()
        if "instance_loss" not in self.trainer_config:
            return nnunet_loss
        kwargs = dict(self.trainer_config["instance_loss"])
        loss_class = get_class(kwargs.pop("_target_"))
        if issubclass(loss_class, (VectorizedCCLoss, VectorizedBiCCLoss)):
            kwargs["sampling"] = self.configuration_manager.spacing
            kwargs["patch_size"] = self.configuration_manager.patch_size
        instance_loss = LesionVsRest(loss_class(**kwargs))
        loss = LesionLoss(nnunet_loss.loss, instance_loss, nnunet_loss.weight_factors)
        return loss.to(self.device)

    def filter_inputs(self, data):
        """`rand_filter` on each sample of a training batch, on the GPU, with the slice axis last while filtering.

        data: (B, C, *spatial) batch from nnU-Net's dataloader
        -> filtered batch on self.device, same layout
        """
        data = data.to(self.device, non_blocking=True)
        slice_axis_last = torch.movedim(data, self.slice_axis, -1)
        filtered = []
        with torch.no_grad():
            for image in slice_axis_last:
                filtered.append(self.rand_filter({"image": image})["image"])
        return torch.movedim(torch.stack(filtered), -1, self.slice_axis)

    def train_step(self, batch):
        if self.rand_filter is not None:
            batch['data'] = self.filter_inputs(batch['data'])
        return super().train_step(batch)


def register_config_trainers():
    """Define `nnUNetTrainer_<name>` in this module for every `segmentation/conf/nnunet_trainer/<name>.yaml`, so that
    `-tr`, prediction and continued training find it by name."""
    for config_path in sorted((CONF_DIR / "nnunet_trainer").glob("*.yaml")):
        config = yaml.safe_load(config_path.read_text())
        name = f"nnUNetTrainer_{config_path.stem}"
        globals()[name] = type(name, (nnUNetTrainerAblation,), {"trainer_config": config})


register_config_trainers()
