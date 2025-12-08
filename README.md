# 🧠 Multimodal Sarcasm Detection using Attention Mechanism

**Author:** Kshitijaa Aigalikar  
**Degree:** MSc Data Science  
**Publication:** International Journal of Innovative Research in Computer and Communication Engineering (IJIRCCE), May 2025  

---

## 📌 Overview

Sarcasm detection is a challenging Natural Language Processing (NLP) problem where the intended meaning of text differs from its literal interpretation. Text-only approaches often fail to capture sarcasm accurately, as sarcasm is highly contextual and frequently influenced by visual cues.

This project presents a **multimodal sarcasm detection system** that integrates **textual and visual information** using an **attention-based deep learning architecture**, resulting in improved sarcasm classification performance.

---

## 🎯 Objectives

- Detect sarcasm using **combined text and image inputs**
- Capture inter-modal dependencies using an **attention mechanism**
- Build an **end-to-end deep learning pipeline**
- Provide **real-time sarcasm detection** via a Streamlit web application

---

## 🧩 Dataset

The dataset used in this project includes:
- Text captions
- Corresponding images
- Binary sarcasm labels (sarcastic / non-sarcastic)

> ⚠️ Due to size and usage restrictions, the dataset is not uploaded to this repository.

### Dataset Structure
# Multimodal Sarcasm Detection Using Attention Mechanism

This project aims to detect sarcasm in conversations using multimodal data: **text (BERT)**, **audio (MFCC)**, and **visual (ResNet)** features, combined using **attention mechanisms**.

## 🧠 Technologies Used
- Python
- PyTorch
- Transformers (BERT)
- ResNet
- Librosa (for MFCC extraction)

## 📁 Dataset
- [MUStARD Dataset](https://github.com/dair-iitd/MUStARD)

## 🧪 Project Overview
- Extracted BERT embeddings from utterances
- Extracted ResNet features from video frames
- Extracted MFCC features from audio
- Used an attention mechanism to fuse all 3 modalities
- Final model achieved **83% precision**

## 🛠️ How to Run
1. Clone the repo  
   `git clone https://github.com/kshitijaa0802/Sarcasm_detection`
2. Install requirements  
   `pip install -r requirements.txt`
3. Run training  
   `python train.py` 

## 📊 Results
- Validation Precision: 83%
- Tools: PyTorch, HuggingFace Transformers

## 📚 Publication
This project was also published in IJIRCCE:  
DOI:10.15680/IJIRCCE.2025.1305250

## 🔗 Contact
Developed by Kshitijaa Aigalikar  
[LinkedIn](https://www.linkedin.com/in/kshitijaa-aigalikar)
