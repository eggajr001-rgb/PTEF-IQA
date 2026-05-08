# PTEF-IQA: Perceptual Tiered Enhancement Framework for No-Reference Image Quality Assessment

> **📌 Note for Reviewers & Readers**
> 
> To protect intellectual property and comply with the double-blind peer review policy, this repository currently serves as an **Inference-Only Demo**. The purpose of this release is to allow reviewers and researchers to verify the predictive behavior, architectural composition, and decoupled calibration mechanism of the proposed PTEF framework without compromising anonymity.
> 
> Upon formal acceptance of the manuscript, the complete open-source repository will be released, including:
> - the full training pipeline,
> - automated multi-seed evaluation scripts,
> - dataset partitioning logs,
> - and the complete set of pre-trained weights.

---

## 📖 Methodology Motivation & Paradigm Formulation

Existing No-Reference Image Quality Assessment (NR-IQA) frameworks predominantly rely on a coupled optimization paradigm, where universal perceptual representation learning and dataset-specific distribution fitting are jointly entangled within a single end-to-end optimization process.

Such coupling often leads to an optimization dilemma: improving adaptation to specific distortion distributions may compromise the model's cross-dataset transferability and general perceptual robustness.

To address this limitation, **PTEF reformulates NR-IQA as a decoupled perceptual estimation and residual calibration problem**, consisting of three complementary components:

### Phase I — Perceptual Prior Injection (Module A: HPSM)
HPSM incorporates Human Visual System (HVS) spatial sensitivity priors into the self-attention mechanism, enhancing localized distortion perception and structural response consistency.

### Phase I — Semantic Consistency Constraint (Module B: SCSA)
SCSA employs cross-layer self-distillation to mitigate representation drift between shallow textural responses and deep semantic representations under complex distortions.

### Phase II — Non-Intrusive Residual Calibration (Module C2: SRA)
**Decoupled Logic:** SRA does not modify the universal perceptual representations learned in Phase I. Instead, it performs lightweight residual calibration on the predicted quality score through a **Stop-Gradient-isolated adaptation pathway**, enabling dataset-aware refinement without perturbing backbone feature optimization.

---

## 🏆 Evaluation Protocol & Experimental Rigor

The evaluation protocol described in the manuscript is designed to ensure reproducibility, statistical stability, and fair comparison against prior methods.

- **Content-Disjoint Split:** For synthetic IQA datasets, data partitioning follows reference-image-level separation to avoid content leakage between training and testing subsets.
- **Multi-Seed Evaluation:** All reported performances are averaged over multiple independent runs with different random seeds to reduce statistical fluctuation.
- **Protocol-Consistent Reproduction:** Baseline methods were reproduced under the identical five-crop evaluation protocol and consistent data partition settings.

---

## 📁 Repository Structure

```text
PTEF_Demo_Code/
├── checkpoints/                  # Directory for downloaded .pt weights
├── models/
│   ├── maniqa.py                 # Core decoupled architecture
│   └── swin.py                   # Swin backbone integrated with HPSM & SCSA
├── test_images/
│   ├── i15_01_5.bmp              # Sample distorted image 1
│   └── i06_22_1.bmp              # Sample distorted image 2
├── utils/
│   └── hpsm_utils.py             # HVS masking and gradient response utilities
├── predict_one_image.py          # Inference script (Five-Crop protocol)
└── requirements.txt              # Minimal environment dependencies
```

---

## 🚀 Quick Start

### 1. Environment Setup
The code has been tested on Python 3.9+ and PyTorch 2.0+. Install dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights
Due to GitHub file size limitations, representative pre-trained weights are hosted on Baidu Netdisk.

* **Download Link / 下载链接:** [Baidu Netdisk (百度网盘)](https://pan.baidu.com/s/1_pkRnvpVR32h-QZqEkkuJg?pwd=1234)
* **Extraction Code / 提取码:** `1234`

Please download and place `ptef_tid2013_A+B+C.pt` inside the `./checkpoints/` directory.

### 3. Run Inference

```bash
python predict_one_image.py \
    --img_path "test_images/i15_01_5.bmp" \
    --ckpt_path "checkpoints/ptef_tid2013_A+B+C.pt"
```

---

## 📈 Expected Output

The inference script explicitly reports both the Phase I perceptual score and the Phase II residual calibration response:

```text
---------------------------------------------------------
 PTEF 预测结果 (基于 5-Crop 测试)
---------------------------------------------------------
 Phase I  基础感知分数 (S_base) : 0.5033
 Phase II SRA补偿幅度 (Δs)      : 0.1459
          实际校准值 (σ*Δs)     : +0.0146 (σ=0.1)
 - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 模型最终输出质量分 (S_final)   : 0.5179
---------------------------------------------------------
```

This output illustrates the decoupled inference behavior of the framework:

`S_final = S_base + σ · Δs`

where:
* **`S_base`** represents the universal perceptual quality estimation learned in Phase I,
* **`Δs`** represents the residual calibration predicted by the SRA module.

---

## 🔒 License
This repository will be released under the MIT License upon formal publication of the manuscript.
