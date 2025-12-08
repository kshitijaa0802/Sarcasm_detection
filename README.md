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
data/
└── mustard/
- [MUStARD Dataset](https://github.com/dair-iitd/MUStARD)

---

## 🧠 Model Architecture

- **Text Encoder:** BERT-based textual feature extractor  
- **Visual Encoder:** CNN-based visual feature extractor  
- **Audio Encoder:** MFCC-based audio feature extraction  
- **Fusion Strategy:** Attention-based multimodal fusion  
- **Classifier:** Fully connected neural network  

The attention mechanism enables the model to dynamically emphasize the most informative modality (text, visual, or audio) during sarcasm prediction.

---

## ⚙️ Project Workflow

1. Load MuSTARD dataset  
2. Extract features:
   - Textual embeddings (BERT)
   - Visual features (CNN)
   - Audio features (MFCC)
3. Fuse multimodal features using attention  
4. Train and evaluate the model  
5. Perform real-time inference using Streamlit  

---

## 📁 Project Structure


---

## 📊 Results

| Metric     | Score |
|------------|-------|
| Accuracy   | **XX%** |
| Precision | **XX** |
| Recall    | **XX** |
| F1-Score  | **XX** |

> Detailed evaluation results are available in the `results/` directory.

---

## 🚀 How to Run

1️⃣ Clone the repository
```bash
git clone https://github.com/kshitijaa0802/Sarcasm_detection.git
cd Sarcasm_detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit Application
cd app
streamlit run streamlit_app.py

The application allows users to input text, upload visual context, and analyze audio cues to detect sarcasm.

🖥️ Streamlit Application Note

GitHub viewers can only see the source code

To use the application, users must:

Run it locally
OR

Access a deployed Streamlit link (if available)

🔮 Future Work

Multilingual and cross-cultural sarcasm detection

Video-level multimodal sarcasm analysis

Transformer-based multimodal fusion

Cloud deployment for public access

📜 Publication

This work has been published in IJIRCCE (May 2025)
📄 DOI: ADD_DOI_HERE

📄 License

This project is licensed under the MIT License.

🤝 Acknowledgements

MuSTARD dataset creators

Research community in multimodal learning

Open-source deep learning and audio processing libraries

## 🔗 Contact
Developed by Kshitijaa Aigalikar  
[LinkedIn](https://www.linkedin.com/in/kshitijaa-aigalikar)
