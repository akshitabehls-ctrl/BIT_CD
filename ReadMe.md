
# Bitemporal Image Transformer for Remote Sensing Change Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-LEVIR--CD-green.svg)](https://justchenhao.github.io/LEVIR/)

**Author:** Akshita Behl
**Institution:** The LNM Institute of Information Technology (LNMIIT)

---

## 📖 Overview

This project presents **BIT-CD**, a state-of-the-art deep learning approach for detecting building changes in high-resolution satellite imagery.

Traditional change detection methods often struggle with "pseudo-changes"—irrelevant variations caused by different lighting conditions, seasons, or sensor noise over time. Our solution addresses this by combining the local feature extraction power of Convolutional Neural Networks (CNNs) with the global context modeling capabilities of Transformers.

We evaluated our model on the challenging **LEVIR-CD dataset**, achieving competitive results that demonstrate robustness against complex environmental conditions.

### Key Achievements
| Metric | Score |
| :--- | :--- |
| **IoU (Intersection over Union)** | **77.62%** |
| **F1-Score** | **87.40%** |

---

## 🧠 Methodology & Architecture

Our approach uses a hybrid **Siamese Encoder-Bottleneck-Decoder** architecture designed specifically to suppress false positives in unchanged regions.

### Key Innovations

1.  **Siamese ResNet50 Backbone:** We extract multi-scale features ($F_{C2}, F_{C3}, F_{C4}$) from both pre-change and post-change images using weight-sharing ResNet50 encoders.
2.  **Bitemporal Image Transformer (BIT):** A Transformer bottleneck processes high-level features to model long-range spatio-temporal context across the two timepoints.
3.  **Difference-Aware UNet Decoder:** This is our critical contribution. Instead of standard feature concatenation, we explicitly calculate the **absolute difference** between temporal features ($|F_A - F_B|$) at every decoder level. This mechanism mathematically suppresses features from static objects, forcing the network to focus only on actual changes.

<img width="1624" height="891" alt="image" src="https://github.com/user-attachments/assets/2d4fd7de-47d6-4ac2-8ed7-8d338ea4574b" />

---

## 🛠️ Setup and Installation

### Prerequisites
* Linux Server (Recommended, e.g., NVIDIA DGX)
* Python 3.8+
* NVIDIA GPU with CUDA support

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/[REPO_NAME].git
    cd [REPO_NAME]
    ```

2.  Install dependencies:
    ```bash
    pip install torch torchvision opencv-python-headless matplotlib tqdm pillow numpy
    ```

---

## 💾 Dataset Preparation

This project uses the **LEVIR-CD** dataset.

1.  **Download** the original LEVIR-CD dataset (1024x1024 images).
2.  **Organize** the raw data into a folder (e.g., `LEVIR-CD-HUGE`) containing `train`, `val`, and `test` subfolders.
3.  **Run the preparation script** to crop the large images into 256x256 patches. This step is crucial for preventing data starvation and fitting images into GPU memory.

```bash
python prepare_data.py --input_dir /path/to/LEVIR-CD-HUGE --output_dir /path/to/LEVIR-CD-256
````

-----

## 🚀 Usage

### 1\. Training

To train the model from scratch using the configurations that yielded our best results:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
--data_dir /path/to/LEVIR-CD-256 \
--checkpoint_dir checkpoints/ \
--bs 16 \
--lr 5e-5 \
--epochs 200
```

*Key hyperparameters: Learning Rate `5e-5`, Batch Size `16`, Hybrid Loss (CrossEntropy + Dice).*

### 2\. Inference & Evaluation

To generate predictions using trained weights. Our prediction pipeline includes **Test-Time Augmentation (TTA)** (averaging predictions across flips/rotations) and **morphological post-processing** for refined boundaries.

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
--data /path/to/LEVIR-CD-256 \
--model_path checkpoints/best_bit.pth \
--output outputs/ \
--num_samples 50
```

-----

## 📂 Project Structure

```
├── checkpoints/        # Saved model weights (.pth files)
├── outputs/            # Generated prediction images
├── dataset.py          # Custom PyTorch Dataset with advanced augmentations
├── model.py            # Complete BIT-CD architecture (ResNet+Transformer+DiffUNet)
├── predict.py          # Inference script with TTA and post-processing
├── prepare_data.py     # Script for cropping 1024x1024 data to 256x256 patches
├── train.py            # Main training loop with logging and validation
└── README.md           # Project documentation
```

-----

## 📄 Reference

You can access the full project report here:

**[BIT_CD_MiniProject_Report.pdf](https://github.com/akshitabehls-ctrl/BIT_CD/blob/main/BIT_CD_MiniProject_Report.pdf)**

### Acknowledgements

We thank the authors of the original LEVIR-CD dataset and the foundational research on Bitemporal Image Transformers.
