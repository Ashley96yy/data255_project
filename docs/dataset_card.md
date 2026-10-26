# Oxford-IIIT Pet Dataset Card

## 1. Dataset Overview

We use the Oxford-IIIT Pet Dataset for our semantic segmentation project. The dataset contains images of cats and dogs from 37 breeds, with corresponding segmentation masks.

Our task is to classify each pixel into one of three classes: foreground, background, or ambiguous border.

The dataset was downloaded and inspected using Torchvision.

## 2. Data Source and License

- **Dataset:** Oxford-IIIT Pet Dataset
- **Source:** https://www.robots.ox.ac.uk/~vgg/data/pets/
- **License:** CC BY-SA 4.0
- **Download method:** `torchvision.datasets.OxfordIIITPet`
- **Data format:** RGB images and pixel-level segmentation masks

The dataset is publicly available. We use it for academic research and will cite the original dataset source.

## 3. Dataset Size and Splits

The dataset contains 7,349 images in total.

| Split | Number of Images |
|---|---:|
| Official Trainval | 3,680 |
| Official Test | 3,669 |
| Total | 7,349 |

We divided the official trainval set into training and validation sets using an 80/20 split, stratified by pet breed with random seed 42.

During data cleaning, seven images with invalid segmentation masks were excluded from training.

The final split is:

| Split | Number of Images |
|---|---:|
| Training | 2,937 |
| Validation | 736 |
| Test | 3,669 |
| Excluded | 7 |

The cleaned split is saved in `configs/train_val_split_clean.csv`.

We will use the same split for all experiments. The official test set will be reserved for final evaluation.

## 4. Input Images and Segmentation Labels

Each input is an RGB image with a corresponding segmentation mask.

The original segmentation masks contain three labels:

| Original Label | Class | Training Label |
|---|---|---|
| 1 | Foreground | 0 |
| 2 | Background | 1 |
| 3 | Ambiguous Border | 2 |

The labels will be converted from 1–3 to 0–2 when preparing the data for PyTorch training.

The ambiguous border class represents uncertain regions around the pet and is not an exact object contour.

## 5. Class Distribution

We calculated the pixel distribution using all 3,680 images in the original trainval split.

| Class | Pixel Percentage |
|---|---:|
| Foreground | 29.17% |
| Background | 59.88% |
| Ambiguous Border | 10.95% |

The background class contains the most pixels, while the ambiguous border class contains the fewest.

This difference motivates our investigation of Dice Loss and class-weighted Cross-Entropy.

These percentages were calculated before removing the seven invalid masks. If class weights are needed during training, they will be calculated using only the cleaned training set.

## 6. Data Quality and Cleaning

We checked the image dimensions, segmentation mask dimensions, and label values during the initial data audit.

No unexpected label values or image-mask size mismatches were found.

However, further inspection identified seven images where pets were clearly visible, but the corresponding masks contained only background pixels.

The affected dataset indices are:

`1404, 2407, 2410, 2437, 2833, 2889, 3246`

We excluded these images from training because the masks would provide incorrect supervision.

The original data was not modified. The exclusions are recorded in the cleaned split CSV.

We also inspected representative images and masks to understand the variation in pet size, background complexity, and ambiguous border regions.

## 7. Preprocessing and Augmentation

For the initial baseline experiments, we plan to use the following preprocessing steps:

- Resize images and masks to 256 × 256.
- Convert RGB images to floating-point tensors and normalize pixel values.
- Convert segmentation labels from 1–3 to 0–2.
- Use bilinear interpolation for images and nearest-neighbor interpolation for masks.
- Apply random horizontal flipping during training.
- Use deterministic preprocessing for validation and testing.

The final preprocessing settings will be documented in the training configuration.

## 8. Data Leakage Prevention

We use a fixed train/validation split and preserve the official test set.

All experiments will use the same data partitions. Validation data will be used for checkpoint selection and hyperparameter tuning, while the test set will only be used for final evaluation.

Any data-dependent class weights will be calculated using the cleaned training set only.

We will also check for possible duplicate or closely related images across the splits.

## 9. Limitations and Intended Use

The dataset contains only cats and dogs, so our results may not generalize to other segmentation tasks.

The ambiguous border annotations are not exact object boundaries. Therefore, the IoU of the border class should not be interpreted as a direct measurement of precise contour accuracy.

We also identified seven invalid segmentation masks during the training-data audit. Other annotation issues may still exist.

The dataset will be used to study how different loss functions affect U-Net segmentation performance, especially for classes with fewer pixels.

## 10. Reproducibility

The dataset audit was performed using `notebooks/data_audit.ipynb`.

The relevant files include:

- `configs/train_val_split_clean.csv` — cleaned training and validation split
- `results/data_audit/data_audit_summary.json` — initial dataset statistics
- `results/data_audit/class_distribution.png` — pixel distribution
- `results/data_audit/representative_samples.png` — sample images and masks
- `results/data_audit/data_quality_issues.csv` — excluded sample records

The original dataset can be downloaded through Torchvision. Raw dataset files are not included in the GitHub repository.
