# CIFAR-10 Image Classification & FastAPI Inference App

An end-to-end deep learning project built for the Everquint AI Engineer Interview Assessment. This repository features a custom Convolutional Neural Network (CNN) trained from scratch on the CIFAR-10 dataset, enhanced with advanced training utilities, and integrated with a FastAPI backend for real-time inference.

---

## 📂 Project Structure

```text
cifar10-classifier/
│
├── train_and_evaluate.py   # Training pipeline, CNN architecture, and evaluation logic
├── app.py                  # FastAPI application for model inference
├── best_cifar_model.pth    # Saved best model weights checkpoint
├── metrics_curve.png       # Training & testing loss/accuracy curves visualization
└── requirements.txt        # Project dependencies
```

---

## 🚀 1. Training Setup & Part 2 Challenge

The training pipeline (`train_and_evaluate.py`) fulfills all Core Coding requirements and integrates **Part 2 - Option B (Training Utilities)**:
* **Data Pipeline**: Uses `torchvision.datasets.CIFAR10` with standard data augmentations (`RandomCrop`, `RandomHorizontalFlip`), normalization, and efficient batching via `DataLoader` (batch size: 128).
* **Reproducibility**: Global random seeds (`set_seed(42)`) are fixed across Python, NumPy, and PyTorch.
* **Part 2 Enhancements (Option B)**:
  * **Early Stopping**: Monitors validation loss and automatically saves the best model checkpoint (`best_cifar_model.pth`) while preventing overfitting.
  * **Gradient Clipping**: Stabilizes backpropagation using `torch.nn.utils.clip_grad_norm_`.
  * **Learning Rate Scheduling**: Utilizes `ReduceLROnPlateau` to dynamically adjust the learning rate.

---

## 🧠 2. Model Design Decisions (`CustomCNN`)

A custom CNN was built from scratch without pre-trained weights, structured to balance efficiency and performance on $32 \times 32$ color images:
* **Feature Extractor**: 2 convolutional blocks featuring `Conv2d` layers (expanding channels from 3 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128), `BatchNorm2d` for internal covariate shift reduction, `ReLU` activations, `MaxPool2d` spatial downsampling, and `Dropout (0.25)` for regularization.
* **Classification Head**: Flatten layer followed by a dense `Linear` layer (512 units), `BatchNorm1d`, `ReLU`, `Dropout (0.5)`, and a final output layer mapping to 10 CIFAR-10 classes.

---

## 📊 3. Evaluation & Performance

* **Metrics Curves**: Training and validation loss/accuracy metrics are plotted and saved locally as `metrics_curve.png`.
* **Misclassified Analysis**: The evaluation script identifies and captures sample misclassified images to analyze model confusion across classes.

---

## ⚙️ 4. How to Run the Code

### **Step 1: Clone the Repository & Install Dependencies**
```bash
git clone https://github.com/RajaRajan71/cifar10-classifier.git
cd cifar10-classifier
pip install -r requirements.txt
```

### **Step 2: Train the Model (Optional)**
To retrain the model and generate a new checkpoint and curves plot:
```bash
python train_and_evaluate.py
```

### **Step 3: Run the FastAPI App locally**
To start the inference API server:
```bash
uvicorn app:app --reload --port 8000
```
Open your browser and navigate to `http://127.0.0.1:8000/docs` to test image predictions interactively via the Swagger UI.