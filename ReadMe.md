# **BIT-CD: Bitemporal Image Transformer for Change Detection**

This repository implements **BIT-CD**, a hybrid deep learning framework for detecting structural changes between pairs of high-resolution remote sensing images. The model integrates **CNN-based spatial representation learning** with **Transformer-based temporal reasoning**, supported by a difference-aware fusion mechanism.

---

## **Abstract**

Change detection in high-resolution remote sensing imagery is essential for applications such as urban development monitoring, disaster assessment, and land-use analysis.
BIT-CD combines a **ResNet50 backbone**, a **Bitemporal Image Transformer (BIT)**, and a **UNet-style decoder** with **Difference-Aware Feature Fusion** to better capture structural changes while suppressing pseudo-changes caused by illumination and seasonal variations.
Evaluated on the LEVIR-CD dataset, the model achieves:

* **IoU:** 77.62%
* **F1-Score:** 87.40%

---

## **Architecture**

BIT-CD consists of three core components:

### **1. Spatial Feature Encoder (ResNet50)**

Extracts multi-scale spatial features from both temporal images:

* ( F_{C2} )
* ( F_{C3} )
* ( F_{C4} )

### **2. Bitemporal Image Transformer (BIT)**

Captures long-range temporal dependencies and aligns context between image pairs.

### **3. UNet-Style Decoder with Difference-Aware Skip Connections**

Integrates absolute feature differences
[
|F_A - F_B|
]
into the decoder pathway to highlight real changes and suppress unchanged areas.

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/yourusername/BIT_CD_Project.git
cd BIT_CD_Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Dataset Preparation**

This project uses the **LEVIR-CD** dataset with 256×256 patching.

Steps:

1. Download the LEVIR-CD dataset.
2. Generate patches using:

```bash
python prepare_data.py
```

This converts 1024×1024 image tiles into 256×256 training and validation patches.

---

## **Training**

Train the model with:

```bash
python train.py --epochs 200 --bs 16 --lr 5e-5
```

Checkpoints and logs will be stored automatically.

---

## **Inference**

Run inference on sample image pairs:

```bash
python predict.py --model_path checkpoints/best_bit.pth --num_samples 20
```

Output masks will be saved in:

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
│── src/
│── train.py
│── predict.py
│── prepare_data.py
│── requirements.txt
└── README.md
```

---

## **License**

This repository is released under the **MIT License**.
