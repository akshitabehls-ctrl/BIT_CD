# **BIT-CD: Bitemporal Image Transformer for Change Detection**

This repository implements **BIT-CD**, a deep learning architecture designed for structural change detection in high-resolution remote sensing imagery. The approach combines convolutional and transformer-based representations with a difference-aware fusion mechanism to enhance temporal feature alignment.

---

## **Abstract**

Change detection in remote sensing imagery is critical for applications such as urban expansion monitoring, disaster assessment, and infrastructure analysis.
BIT-CD integrates:

* a **ResNet50** feature extractor,
* a **Bitemporal Image Transformer (BIT)** for temporal context modeling, and
* a **UNet-style decoder** enhanced with **Difference-Aware Feature Fusion**.

This design improves robustness against pseudo-changes caused by illumination and seasonal variations.
On the LEVIR-CD dataset, BIT-CD achieves:

* **IoU:** 77.62%
* **F1-Score:** 87.40%

---

## **Architecture Overview**

BIT-CD consists of three main components:

### **1. Spatial Feature Encoder (ResNet50)**

Extracts multi-scale spatial features from both temporal images at the following stages:

* ( F_{C2} )
* ( F_{C3} )
* ( F_{C4} )

### **2. Bitemporal Image Transformer (BIT)**

Captures long-range temporal relationships and aligns contextual information between the two temporal inputs.

### **3. UNet-Style Decoder with Difference-Aware Skip Connections**

Skip connections incorporate absolute feature differences:

[
\left| F_A - F_B \right|
]

This emphasizes genuine structural changes while reducing the influence of unchanged regions or illumination variations.

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/yourusername/BIT_CD_Project.git
cd BIT_CD_Project
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## **Dataset Preparation**

This project uses the **LEVIR-CD** dataset. Images must be cropped into 256×256 patches for training.

To generate patches:

```bash
python prepare_data.py
```

The script processes 1024×1024 images into training and validation tiles.

---

## **Training**

Train the BIT-CD model using:

```bash
python train.py --epochs 200 --bs 16 --lr 5e-5
```

All checkpoints and logs are stored in the `checkpoints/` and corresponding log files.

---

## **Inference**

Run inference on test samples:

```bash
python predict.py --model_path checkpoints/best_bit.pth --num_samples 20
```

Predictions are saved in:

```
outputs_final/
```

---

## **Results (LEVIR-CD)**

| Metric    | Score  |
| --------- | ------ |
| IoU       | 77.62% |
| F1-Score  | 87.40% |
| Precision | 89.10% |
| Recall    | 85.60% |

---

## **Project Structure**

```
BIT_CD/
│── checkpoints/
│── outputs_final/
│── dataset/
│── src/
│── train.py
│── predict.py
│── prepare_data.py
│── requirements.txt
└── README.md
```

---

## **License**

This project is released under the **MIT License**.
