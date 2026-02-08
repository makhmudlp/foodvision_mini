# FoodVision Mini 🍕🥗🍔

FoodVision Mini is a lightweight food image classification project that compares **EfficientNet-B2** and **Vision Transformer (ViT)** models, with a focus on **real-world inference speed, accuracy, and deployment**.

The project demonstrates the full machine learning workflow — from data preparation and training to benchmarking and deployment — using **PyTorch** and **Hugging Face Spaces**.

---

## 🚀 Live Demo

👉 Hugging Face Space: https://huggingface.co/spaces/makhmudlp/foodvision_mini

Upload an image and get an instant prediction from the deployed model.

---

## 📌 Project Overview

FoodVision Mini classifies images into **3 food categories**.

The main goals of this project are to:

- Train a small but effective image classifier
- Compare **EfficientNet-B2 vs Vision Transformer (ViT)**
- Measure inference speed under **CPU, single-image conditions**
- Understand why benchmark results differ across environments
- Deploy a real, usable model using **Gradio on Hugging Face Spaces**

This project intentionally focuses on **practical ML engineering tradeoffs**, not just model size.

---

## 🧠 Models Evaluated

### EfficientNet-B2
- Convolutional neural network (CNN)
- Small model size (~30 MB)
- Strong accuracy
- Slower single-image CPU inference in this setup

### Vision Transformer (ViT)
- Transformer-based architecture
- Larger model size (~327 MB)
- Higher accuracy
- Faster single-image CPU inference due to optimized matrix operations

---

## 📊 Results

| Model | Test Accuracy | Model Size | CPU Time / Image |
|------|--------------|------------|------------------|
| EfficientNet-B2 | 95.97% | ~30 MB | ~138 ms |
| Vision Transformer (ViT) | 98.47% | ~327 MB | ~52 ms |

**Key insight:**  
Despite being much larger, ViT achieved faster CPU inference in this deployment scenario due to better utilization of optimized linear algebra operations. This highlights how **model size alone is not a reliable predictor of inference speed**.

---

## 🚀 Deployment

- Framework: **Gradio**
- Platform: **Hugging Face Spaces**
- Hardware: **CPU (Free tier)**
- Inference mode: **Single image**

The **EffNetB2** model was selected for deployment due to its faster inference on HuggingFace Spaces.

📌 **Note:** Only the `deploying/foodvision_mini/` directory is required for deployment.

---

## 📁 Repository Structure

```text
.
├── deploying/foodvision_mini/
│   ├── app.py
│   ├── model.py
│   ├── requirements.txt
│   └── (files used for Hugging Face Spaces)
│
├── going_modular/
│   └── Training and experimentation utilities
│
├── helper_functions.py
│   └── Shared helper functions used in notebooks
│
├── model_deployment.ipynb
│   └── End-to-end project walkthrough
│
├── README.md
└── LICENSE
```
