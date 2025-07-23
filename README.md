# Multimodal Sarcasm Detection Using Attention Mechanism

This project aims to detect sarcasm in conversations using multimodal data: **text (BERT)**, **audio (MFCC)**, and **visual (ResNet)** features, combined using **attention mechanisms**.

## 🧠 Technologies Used
- Python
- PyTorch
- Transformers (BERT)
- OpenCV / ResNet
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
   `python train.py` *(or Jupyter notebook)*

## 📊 Results
- Validation Precision: 83%
- Tools: PyTorch, HuggingFace Transformers

## 📚 Publication
This project was also published in IJIRCCE:  
[DOI link here if available]

## 🔗 Contact
Developed by Kshitijaa Aigalikar  
[LinkedIn](https://www.linkedin.com/in/kshitijaa-aigalikar)
