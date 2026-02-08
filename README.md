# FoodVision Mini 🍕🥗🍔

FoodVision Mini is a lightweight food image classification project that compares Vision Transformers (ViT) and EfficientNet for fast CPU-based inference and deployment. And Deploys the best option to HugginFace spaces

## 🚀 Live Demo

👉 Try the model here: https://huggingface.co/spaces/makhmudlp/foodvision_mini

## 📌 Project Overview

This project classifies images into **3 food categories** using deep learning.
The main goal is to:

- Train a small but effective image classifier
- Compare **EfficientNet-B2 vs Vision Transformer (ViT)**
- Benchmark inference speed under real deployment conditions
- Deploy the final model using **Gradio on Hugging Face Spaces**
## 🧠 Models

Two architectures were evaluated:

- **EfficientNet-B2**
  - Lightweight CNN
  - Small model size (~30 MB)
  - Slower single-image CPU inference

- **Vision Transformer (ViT)**
  - Larger model (~327 MB)
  - Dense matrix operations
  - Faster single-image CPU inference in this setup

## 🚀 Deployment

- Framework: **Gradio**
- Platform: **Hugging Face Spaces**
- Hardware: **CPU (Free tier)**
- Inference mode: Single image

The ViT model was selected for deployment due to its higher accuracy and lower latency under CPU inference.

## 🛠 Tech Stack

- Python
- PyTorch
- Torchvision
- Gradio
- Hugging Face Spaces
- Git LFS

