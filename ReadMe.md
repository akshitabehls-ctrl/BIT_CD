BIT-CD: Bitemporal Image Transformer for Change Detection

📌 Abstract

Change detection in high-resolution remote sensing imagery is a critical task for urban planning and disaster monitoring. This project implements a hybrid deep learning architecture combining a ResNet50 backbone with a Bitemporal Image Transformer (BIT) and a UNet-style decoder. We introduce a novel Difference-Aware Feature Fusion mechanism to suppress pseudo-changes caused by seasonal variations and lighting differences. Evaluated on the LEVIR-CD dataset, our method achieves an Intersection over Union (IoU) of 77.62% and an F1-Score of 87.40%.

🧠 Architecture

The model utilizes a hybrid CNN-Transformer architecture:

Backbone: ResNet50 (Pretrained) extracting multi-scale features ($F_{C2}, F_{C3}, F_{C4}$).

Context: Bitemporal Image Transformer (BIT) to model global context.

Decoder: Custom UNet-style decoder with Absolute Difference Skip Connections ($|F_A - F_B|$) to suppress unchanged features.

🛠️ Installation & Setup

Clone the repository:

git clone [https://github.com/yourusername/BIT_CD_Project.git](https://github.com/yourusername/BIT_CD_Project.git)
cd BIT_CD_Project


Install Dependencies:

pip install -r requirements.txt


📊 Dataset Preparation

The model requires the LEVIR-CD dataset patched into 256x256 crops.

Download the raw LEVIR-CD dataset.

Run the preparation script to crop 1024x1024 images into 256x256 patches:

python prepare_data.py


🚀 Training

To train the model from scratch with GPU acceleration:

python train.py --epochs 200 --bs 16 --lr 5e-5


🖼️ Inference

To generate visual predictions using the trained model:

python predict.py --model_path checkpoints/best_bit.pth --num_samples 20


Results are saved in the outputs/ folder.

📈 Results

Metric

Score

IoU (Intersection over Union)

77.62%

F1-Score

87.40%

Precision

89.1%

Recall

85.6%

📝 License

This project is open-source and available under the MIT License.