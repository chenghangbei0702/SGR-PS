# SGR-PS
Semantic-Geometric-Reliability Learning for  Semi-Supervised Medical Image Segmentation

# SGR-PS: Semantic-Geometric-Reliability Structured Pseudo-Supervision for Semi-Supervised Medical Image Segmentation

This repository contains the official implementation of the paper:

> **Semantic-Geometric-Reliability Learning for Semi-Supervised Medical Image Segmentation**  

SGR-PS reformulates pseudo-labeling as **structured pseudo-supervision**, decomposing unlabeled supervision into three complementary roles:
- **Semantic Grounding**: target identity via a frozen biomedical vision–language model (BiomedCLIP).
- **Geometric Refinement**: controlled prompt-based refinement with a frozen promptable segmentation model (SAM), including Asymmetric Grid Prompt Sampling (AGPS) and Semantic-Uncertainty Selective Correction (SUSC).
- **Reliability Learning**: a learned heteroscedastic uncertainty head that adapts pixel-wise supervision.

All auxiliary components are used **only during training**. Inference relies solely on a lightweight student network.

---


## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+

Main packages: `torch`, `torchvision`, `numpy`, `opencv-python`, `scikit-image`, `pyyaml`, `tqdm`, `tensorboard`.

---

## 📂 Data Preparation

We evaluate on five public datasets:

| Dataset | Modality | Task |
|---------|----------|------|
| Lung | X-ray | Lung segmentation |
| Lung CT | CT | Lung CT segmentation |
| Kvasir-SEG | Endoscopy | Polyp segmentation |
| ISIC 2018 | Dermoscopy | Skin lesion segmentation |
| Breast Tumors | Ultrasound | Breast tumor segmentation |

Please download the datasets and place them under `data/` as follows:

```
data/
├── Lung/
│   ├── train/
│   ├── val/
│   └── test/
├── LungCT/
├── Kvasir/
├── ISIC2018/
└── BreastTumors/
```

Each dataset should contain `images/` and `masks/` subfolders. 

**Pretrained models**:
- BiomedCLIP: download from [https://huggingface.co/ZiyueWang/biomedclip] 
- SAM (ViT-B): download from [https://github.com/facebookresearch/segment-anything] 

---

## 🚀 Training

To train SGR-PS on a specific dataset and labeling ratio:

```bash
python train.py --config configs/lung.yaml --label_ratio 0.1 --gpu 0
```

Example for Lung with 10% labels:

```bash
python train.py --config configs/lung.yaml --label_ratio 0.1
```

---

## 🧪 Testing

To evaluate a trained model:

```bash
python test.py --config configs/lung.yaml --checkpoint checkpoints/lung/0.1/best.pth
```

---

## 🙏 Acknowledgements

We thank the authors of BiomedCLIP, SAM, and the public datasets for their contributions.
