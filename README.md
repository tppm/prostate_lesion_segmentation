# Prostate lesion segmentation

Custom nnU-Net v2 trainers for prostate lesion segmentation on T2, ADC and DWI.

## Install

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
mkdir -p nnunetv2/training/nnUNetTrainer/variants/lesion
cp ../nnunet_trainer/*.py nnunetv2/training/nnUNetTrainer/variants/lesion/
pip install -e .
cd ..
```

## Data

Use the standard nnU-Net dataset format, with:

- **Channels:** `0000` T2, `0001` ADC, `0002` DWI, named `T2`, `ADC` and `DWI` in `dataset.json`
- We resampled ADC and DWI resampled onto T2
- **Labels without BackSplit:** background 0, lesion 1
- **Labels with BackSplit:** background 0, gland 1, lesion 2

## Train and predict

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID -pl nnUNetPlannerResEncM -c 3d_fullres
nnUNetv2_train DATASET_ID 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr TRAINER
nnUNetv2_predict -i INPUT_DIR -o PRED_DIR -d DATASET_ID -c 3d_fullres -p nnUNetResEncUNetMPlans -tr TRAINER -f 0
```

The final model is BackSplit with `nnUNetTrainer_cc_rc_wavelet_sym4`: nnU-Net's CE + Dice, a lesion CC-Dice-CE term, and random sym4 wavelet augmentation.

## Pretrained model

The trained model folder is hosted on Hugging Face at `HF_USER/HF_REPO`. Install the trainers (see [Install](#install)), then download it into your `nnUNet_results` folder:

Install huggingface cli:  https://huggingface.co/docs/huggingface_hub/guides/cli
```bash
hf download HF_USER/HF_REPO --local-dir "$nnUNet_results"
```

This should give `$nnUNet_results/Dataset506_NAME/nnUNetTrainer_cc_rc_wavelet_sym4__nnUNetResEncUNetMPlans__3d_fullres/`. nnU-Net finds it through the `nnUNet_results` environment variable, so predict with:

```bash
nnUNetv2_predict -i INPUT_DIR -o PRED_DIR -d 506 -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainer_cc_rc_wavelet_sym4 -f 0
```

Input files must follow the channel order described in [Data](#data).

## Evaluate

```bash
python -m eval.evaluate --dataset-dir DATASET_DIR --pred-dir PRED_DIR
```

`DATASET_DIR` needs `dataset.json`, `labelsTs/`, and a `cases.csv` with `case_id`, `cohort` and `split` columns. Scores are written to `PRED_DIR/per_case.csv`.

## Notes

The published models were trained on a fork of nnU-Net at this commit. 

## Credits


## Citation

If you used this repo, please cite our paper: 
