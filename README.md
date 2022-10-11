# SuperYOLO
# This code will be completely released after our [article](https://arxiv.org/abs/2209.13351) is received！！！！
⭐ Please wait patiently！！！Thank you！！！⭐ 

If our code is helpful to you, please cite:

`@article{zhang2022superyolo,
  title={SuperYOLO: Super Resolution Assisted Object Detection in Multimodal Remote Sensing Imagery},
  author={Zhang, Jiaqing and Lei, Jie and Xie, Weiying and Fang, Zhenman and Li, Yunsong and Du, Qian},
  journal={arXiv preprint arXiv:2209.13351},
  year={2022}
}` 

## Requirements

```python
pip install -r requirements.txt
```

## Train

### 1. Prepare training data 

- 1.1 In order to realize the SR assisted branch, the input images of the network are downsampled from 1024 x 1024 size to 512 x 512 during the training process. In the test process, the image size is 512 x 512, which is consistent with the input of other algorithms compared.

- 1.2 Download VEDAI data for our experiment from [baiduyun](https://pan.baidu.com/s/1L0SWi5AQA6ZK9jDIWRY7Fg) (code: hvi4).

- 1.3 Note that we transform the labels of the dataset to be horizontal boxes by [transform code](data/transform.py). You shoud run transform.py before training the model.

### 2. Begin to train

```python
python train_up.py --cfg models/SRyolo_noFocus.yaml --super --train_img_size 1024 --hr_input --data data/SRvedai.yaml --ch 4
```

## Test

### Pretrained Checkpoints
You can use our pretrained checkpoints for test process.
Download pre-trained model and put it in [here](https://github.com/icey-zhang/SuperYOLO/tree/main/weights).

| Dataset |Valiadation | Weight | **mAP50** |
|:---:|:---:|:---:|:---:|
| VEDAI | fold1 |[small_EDSR_fold1](https://pan.baidu.com/s/1Lyue-6CAUPn6KnEEOCOppg?pwd=6666)|0.8252 |
| VEDAI | fold2 |[small_EDSR_fold2](https://pan.baidu.com/s/1k5AzGStOq13B2kj_rvZRsA?pwd=6666)| 0.7053 |
| VEDAI | fold3 |[small_EDSR_fold3](https://pan.baidu.com/s/15XgBDlK0qTN5ATZmr4aUGw?pwd=6666)| 0.7241 |
| VEDAI | fold4 |[small_EDSR_fold4](https://pan.baidu.com/s/1C5zBwfndFwHCW833-KM3tA?pwd=6666)| 0.7749 |
| VEDAI | fold5 |[small_EDSR_fold5](https://pan.baidu.com/s/1iqDgtVOqlmcy6uiEtBP_Xg?pwd=6666)| 0.7448 |
| VEDAI | fold6 |[small_EDSR_fold6](https://pan.baidu.com/s/10mqgEWvOPRH3PRiqch7y2A?pwd=6666)| 0.7525 |
| VEDAI | fold7 |[small_EDSR_fold7](https://pan.baidu.com/s/1-viB3Btspl9ZCaXDI_gAow?pwd=6666)| 0.7010 |
| VEDAI | fold8 |[small_EDSR_fold8](https://pan.baidu.com/s/11TCNq4B1dhyfg-iBjdiPmA?pwd=6666)| 0.7662 |
| VEDAI | fold9 |[small_EDSR_fold9](https://pan.baidu.com/s/1LpILxKI1qJHQekVuJQAiQw?pwd=6666) | 0.7374 |
| VEDAI | fold10 | [small_EDSR_fold10](https://pan.baidu.com/s/12ELeCg6rZuUO4YaaXcuJOg?pwd=6666) | 0.7479 |

```python
python test.py --weights runs/train/exp/best.pt --input_mode RGB+IR 
```

## Results

| Method | Modality |  **Car**  | **Pickup** | **Camping** | **Truck** | **Other** | **Tractor** | **Boat** | **Van** | **mAP50** | **Params.** $\downarrow$ | **GFLOPs** $\downarrow$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **YOLOv3** | IR | 80.21 | 67.03 | 65.55 | 47.78 | 25.86 | 40.11 | 32.67 | 53.33 | 51.54 | **61.5351M** | 49.55 |
| **YOLOv3** | RGB | 83.06 | 71.54 | **69.14** | 59.30 | **48.93** | **67.34** | 33.48 | 55.67 | 61.06 | **61.5351M** | 49.55 |
| **YOLOv3** | Multi | **84.57** | **72.68** | 67.13 | **61.96** | 43.04 | 65.24 | **37.10** | **58.29** | **61.26** | 61.5354M | 49.68 |
| **YOLOv4** | IR | 80.45 | 67.88 | 68.84 | 53.66 | 30.02 | 44.23 | 25.40 | 51.41 | 52.75 | **52.5082M** | 38.16 |
| **YOLOv4** | RGB | 83.73 | **73.43** | 71.17 | 59.09 | **51.66** | 65.86 | **34.28** | **60.32** | 62.43 | **52.5082M** | 38.16 |
| **YOLOv4** | Multi | **85.46** | 72.84 | **72.38** | **62.82** | 48.94 | **68.99** | **34.28** | 54.66 | **62.55** | 52.5085M | 38.23 |
| **YOLOv5s** | IR | 77.31 | 65.27 | 66.47 | 51.56 | 25.87 | 42.36 | 21.88 | 48.88 | 49.94 | **7.0728M** | 5.24 |
| **YOLOv5s** | RGB | 80.07 | 68.01 | 66.12 | 51.52 | 45.76 | **64.38** | 21.62 | 40.93 | 54.82 | **7.0728M** | 5.24 |
| **YOLOv5s** | Multi | 80.81 | **68.48** | **69.06** | **54.71** | **46.76** | 64.29 | **24.25** | **45.96** | **56.79** | 7.0739M | 5.32 |
| **YOLOv5m** | IR | 79.23 | 67.32 | 65.43 | 51.75 | 26.66 | 44.28 | 26.64 | 56.14 | 52.19 | **21.0659M** | 16.13 |
| **YOLOv5m** | RGB | 81.14 | 70.26 | 65.53 | 53.98 | **46.78** | **66.69** | **36.24** | 49.87 | 58.80 | **21.0659M** | 16.13 |
| **YOLOv5m** | Multi | **82.53** | **72.32** | **68.41** | **59.25** | 46.20 | 66.23 | 33.51 | **57.11** | **60.69** | 21.0677M | 16.24 |
| **YOLOv5l** | IR | 80.14 | 68.57 | 65.37 | 53.45 | 30.33 | 45.59 | 27.24 | **61.87** | 54.06 | **46.6383M** | 36.55 |
| **YOLOv5l** | RGB | 81.36 | 71.70 | 68.25 | 57.45 | 45.77 | **70.68** | 35.89 | 55.42 | 60.81 | **46.6383M** | 36.55 |
| **YOLOv5l** | Multi | **82.83** | **72.32** | **69.92** | **63.94** | **48.48** | 63.07 | **40.12** | 56.46 | **62.16** | 46.6046M | 36.70 |
| **YOLOv5x** | IR | 79.01 | 66.72 | 65.93 | 58.49 | 31.39 | 41.38 | 31.58 | 58.98 | 54.18 | **87.2458M** | 69.52 |
| **YOLOv5x** | RGB | 81.66 | 72.23 | 68.29 | 59.07 | 48.47 | 66.01 | **39.15** | **61.85** | 62.09 | **87.2458M** | 69.52 |
| **YOLOv5x** | Multi | **84.33** | **72.95** | **70.09** | **61.15** | **49.94** | **67.35** | 38.71 | 56.65 | **62.65** | 87.2487M | 69.71 |
| **SuperYOLO** | IR | 87.90 | 81.39 | 76.90 | 61.56 | 39.39 | 60.56 | 46.08 | **71.00** | 65.60 | **4.8256M** | 16.61 |
| **SuperYOLO** | RGB | 90.30 | 82.66 | 76.69 | 68.55 | 53.86 | 79.48 | 58.08 | 70.30 | 72.49 | **4.8256M** | 16.61 |
| **SuperYOLO** | Multi | **90.86** | **84.35** | **78.11** | **68.11** | **53.26** | **82.33** | **60.95** | 70.94 | **73.61** | 4.8259M | 16.68 |

## Visualization of results

<p align="center"> <img src="Fig/results.png" width="90%"> </p>

## Acknowledgements
This code is built on [YOLOv5 (PyTorch)](https://github.com/ultralytics/yolov5). We thank the authors for sharing the codes.
