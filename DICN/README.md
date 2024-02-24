# DICN
This is the official PyTorch implementation for the paper:

> Automatic Road Extraction with Multi-Source Data Revisited: Complete, Smooth and Discriminative.

## Dataset Preprocessing
For details on how to preprocess the dataset, please refer to [Link](https://github.com/suniique/Leveraging-Crowdsourced-GPS-Data-for-Road-Extraction-from-Aerial-Imagery) and [CMMPNet](https://github.com/liulingbo918/CMMPNet/tree/main).

## Usage
```bash
## train
python train.py \
  --model segtrans \
  --b 4 \
  --lr 2e-4 \
  --epochs 200 \
  --prefix work_dir

## test
python train.py \
  --model segtrans \
  --b 1 \
  --eval predict \
  --weight_load_path xxx.pth
```
