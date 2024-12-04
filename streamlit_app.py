from transformers import AutoTokenizer, AutoModel
from torchvision.models import resnet50
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import streamlit as st
from pathlib import Path
import librosa
import cv2
import tempfile
from torchvision import transforms


class MultimodalSarcasmModel(nn.Module):
    def __init__(self, text_dim, audio_dim, video_dim, context_dim, hidden_dim, output_dim):
        super(MultimodalSarcasmModel, self).__init__()
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim)
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        self.video_fc = nn.Linear(video_dim, hidden_dim)
        combined_dim = hidden_dim * 5
        self.final_fc = nn.Linear(combined_dim, output_dim)

    def forward(self, text, context, audio, utterance_video, context_video):
        text_out = F.relu(self.text_fc(text.float()))
        context_out = F.relu(self.context_fc(context.float()))

        if len(audio.shape) > 2:
            audio = torch.mean(audio, dim=1)
        audio_out = F.relu(self.audio_fc(audio.float()))

        utterance_video_out = F.relu(self.video_fc(utterance_video.float()))
        context_video_out = F.relu(self.video_fc(context_video.float()))

        combined = torch.cat([text_out, context_out, audio_out, utterance_video_out, context_video_out], dim=-1)
        output = self.final_fc(combined)
        return output


def load_model():
    """Load the pretrained model and tokenizer"""
    model_params = {
        'text_dim': 768,
        'context_dim': 768,
        'audio_dim': 12,
        'video_dim': 2048,  # ResNet50 feature dimension
        'hidden_dim': 512,
        'output_dim': 2
    }

    # Initialize model
    model = MultimodalSarcasmModel(**model_params)

    # Load the saved model weights if they exist
    model_path = Path('model_weights.pth')
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Load BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased')

    # Load ResNet model and remove the final classification layer
    resnet = resnet50()
    resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the final FC layer
    resnet.eval()

    return model, tokenizer, bert_model, resnet


def process_text(text, tokenizer, bert_model):
    """Process text input using BERT tokenizer and model"""
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return embeddings


def process_audio(audio_file):
    """Process audio file to extract features"""
    try:
        y, sr = librosa.load(audio_file)
        features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12), axis=1)
        return torch.FloatTensor(features).unsqueeze(0)
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def process_video(video_file, resnet_model):
    """Process video file to extract features using ResNet50"""
    try:
        # Define image transformations
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            temp_path = temp_file.name

        cap = cv2.VideoCapture(temp_path)
        frames_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform frame
            frame_tensor = transform(frame).unsqueeze(0)

            # Extract features
            with torch.no_grad():
                features = resnet_model(frame_tensor)
                frames_features.append(features.squeeze())

        cap.release()

        # Average features across frames
        if frames_features:
            frame_features = torch.stack(frames_features)
            mean_features = torch.mean(frame_features, dim=0)
            return mean_features.unsqueeze(0)
        else:
            raise ValueError("No frames were processed from the video")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None


def main():
    st.set_page_config(page_title="Multimodal Sarcasm Detection", layout="wide")
    st.title("Multimodal Sarcasm Detection")

    # Load model, tokenizer, and BERT model
    model, tokenizer, bert_model, resnet_model = load_model()
    model.eval()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Text Input")
        utterance_text = st.text_area("Enter the utterance:", height=100)
        context_text = st.text_area("Enter the context:", height=100)

        st.subheader("Audio Input")
        audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])

    with col2:
        st.subheader("Video Input")
        utterance_video = st.file_uploader("Upload Utterance Video", type=['mp4', 'avi'])
        context_video = st.file_uploader("Upload Context Video", type=['mp4', 'avi'])

    if st.button("Detect Sarcasm"):
        if utterance_text and audio_file and utterance_video:
            with st.spinner("Processing inputs..."):
                # Process inputs
                text_features = process_text(utterance_text, tokenizer, bert_model)
                context_features = process_text(context_text, tokenizer, bert_model)
                audio_features = process_audio(audio_file)
                utterance_video_features = process_video(utterance_video, resnet_model)
                context_video_features = process_video(context_video, resnet_model)

                if all([text_features is not None,
                        context_features is not None,
                        audio_features is not None,
                        utterance_video_features is not None,
                        context_video_features is not None]):

                    # Make prediction
                    with torch.no_grad():
                        outputs = model(
                            text_features,
                            context_features,
                            audio_features,
                            utterance_video_features,
                            context_video_features
                        )
                        probabilities = F.softmax(outputs, dim=1)
                        prediction = torch.argmax(probabilities, dim=1).item()

                    st.subheader("Results")
                    if prediction == 1:
                        st.warning("🎭 This appears to be SARCASTIC")
                    else:
                        st.success("✨ This appears to be NON-SARCASTIC")

                    st.write("Confidence Scores:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Non-Sarcastic", f"{probabilities[0][0].item():.2%}")
                    with col2:
                        st.metric("Sarcastic", f"{probabilities[0][1].item():.2%}")
                else:
                    st.error("Error processing one or more inputs. Please check the error messages above.")
        else:
            st.error("Please provide all required inputs (text, audio, and video)")


if __name__ == "__main__":
    main()
