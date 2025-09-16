import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import gdown
from model import load_model

# --------------------------
# Google Drive Model Download + Cache
# --------------------------
MODEL_ID = "1tLnKKniWpkAss2Vp7auKuJsorrqtZpVU"
MODEL_PATH = "best_model.pth"
MODEL_SIZE_MB = 200  # approx size of your model

def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < MODEL_SIZE_MB * 1024 * 1024:
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False, fuzzy=True)
    return MODEL_PATH

@st.cache_resource
def load_cached_model():
    model_path = download_model()
    model, device = load_model(model_path, num_classes=7)
    return model, device

# --------------------------
# Load model once
# --------------------------
model, device = load_cached_model()

fer_emotions = ["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.485,0.485], std=[0.229,0.229,0.229])
])

# --------------------------
# Dataset + Emotion → Genre mapping
# --------------------------
df = pd.read_csv(
    "movies_dataset.csv",
    engine='python',
    encoding='utf-8',
    on_bad_lines='skip'
)

emotion_to_genre = {
    "Anger": ["Action", "Crime"],
    "Disgust": ["Comedy"],
    "Fear": ["Horror", "Mystery"],
    "Happy": ["Comedy", "Animation"],
    "Sad": ["Drama", "Biography"],
    "Surprise": ["Adventure", "Fantasy"],
    "Neutral": ["Drama"]
}

# --------------------------
# Helpers
# --------------------------
def predict_emotion(img: Image.Image):
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]
    top_idx = probs.argmax()
    return fer_emotions[top_idx], probs[top_idx]

def recommend_movies(emotion, top_n=5):
    genres = emotion_to_genre.get(emotion, ["Drama"])
    filtered = df[df["main_genre"].apply(lambda x: any(g in x for g in genres))]
    if filtered.empty:
        return ["No movies found for this emotion."]
    return filtered.sample(min(top_n, len(filtered)))["Movie_Title"].tolist()

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎭 Emotion Detection + 🎬 Movie Recommendation")

option = st.radio("Choose input method:", ["Upload Image", "Webcam"])

if option == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
    if file is not None:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        emotion, prob = predict_emotion(img)
        st.success(f"Predicted Emotion: **{emotion}** ({prob:.2f})")

        st.subheader("🎬 Recommended Movies:")
        for m in recommend_movies(emotion, top_n=5):
            st.write(f"- {m}")

elif option == "Webcam":
    st.write("Click **Start** to use your webcam")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not working!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            emotion, prob = predict_emotion(pil_img)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} ({prob:.2f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

