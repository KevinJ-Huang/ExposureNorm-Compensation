# Exposure Normalization and Compensation for Multiple-Exposure Correction (CVPR 2022)

Jie Huang, Yajing Liu, Xueyang Fu, Man Zhou, Yang Wang, Feng Zhao*, Zhiwei Xiong
*Corresponding Author

University of Science and Technology of China (USTC)

## Introduction

This repository is the **official implementation** of the paper, "Exposure Normalization and Compensation for Multiple-Exposure Correction", where more implementation details are presented.

### 0. Hyper-Parameters setting

Overall, most parameters can be set in options/train/train_Enhance_MSEC.yml or options/train/train_Enhance_SICE.yml

### 1. Dataset Preparation

Create a .txt file to put the path of the dataset using 

```python
python create_txt.py
```

### 2. Training

```python
python train.py --opt options/train/train_Enhance_MSEC.yml or train_Enhance_SICE.yml
```


### 3. Inference

set is_training in "options/train/train_Enhance_MSEC.yml or train_Enhance_SICE.yml" as False
set the val:filelist as the validation set. 

then
```python
python train.py --opt options/train/train_Enhance_MSEC.yml or train_Enhance_SICE.yml
```

## Dataset 
MSEC dataset (please refer to https://github.com/mahmoudnafifi/Exposure_Correction)

SICE dataset (I have uploaded it to https://share.weiyun.com/C2aJ1Cti)

## Ours Results

coming soon


## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (hj0117@mail.ustc.edu.cn).

## Cite

```
@inproceedings{huang2022exposure,
  title={Exposure Normalization and Compensation for Multiple-Exposure Correction},
  author={Huang, Jie and Liu, Yajing and Fu, Xueyang and Zhou, Man and Wang, Yang and Zhao, Feng and Xiong, Zhiwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6043--6052},
  year={2022}
}
```
