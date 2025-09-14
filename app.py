import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import MyCnn
import io

# ========== Config ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = MyCnn(num_classes=7)
state_dict = torch.load("best-model.pth", map_location=device)
model.load_state_dict(state_dict)   # ⚠️ function call, assign mat karna

model.to(device)
model.eval()

# Emotions + Mapping
fer_emotions = ["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

emotion_to_genre = {
    "Anger": ["Action", "Thriller"],
    "Disgust": ["Comedy", "Family"],
    "Fear": ["Horror", "Thriller"],
    "Happy": ["Romance", "Comedy"],
    "Sad": ["Drama", "Biography"],
    "Surprise": ["Adventure", "Fantasy"],
    "Neutral": ["Drama", "Documentary"]
}

# Movies Dataset
df = pd.read_csv("movies_dataset.csv")

# Image Transform (⚠️ confirm yehi training time pe tha ya nahi)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Recommend Movies Function
def recommend_movies(genre, top_n=5):
    filtered_df = df[df['main_genre'].str.contains(genre, case=False, na=False)]
    if len(filtered_df) == 0:
        return []
    return filtered_df.sample(min(top_n, len(filtered_df)))['Movie_Title'].tolist()

# Predict Emotion Function
def predict_emotion(img):
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        emotion = fer_emotions[pred_idx]
        genres = emotion_to_genre[emotion]
    return emotion, genres, probs.cpu().numpy()

# ========== Streamlit UI ==========
st.title("🎭 Emotion-based Movie Recommender")

st.sidebar.header("Choose Input Mode")
mode = st.sidebar.radio("Select input source:", ["Upload Image", "Webcam"])

img = None

if mode == "Upload Image":
    uploaded = st.file_uploader("Upload a face image", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

elif mode == "Webcam":
    camera_image = st.camera_input("Take a photo")
    if camera_image:
        img = Image.open(io.BytesIO(camera_image.getvalue())).convert('RGB')
        st.image(img, caption="Captured Image", use_container_width=True)

# ✅ Predict button
if img:
    if st.button("🔮 Predict Emotion & Recommend Movies"):
        emotion, genres, probs = predict_emotion(img)
        st.subheader(f"Detected Emotion: {emotion}")
        st.write(f"Recommended Genres: {genres}")

        movies = []
        for g in genres:
            movies.extend(recommend_movies(g, top_n=3))
        if movies:
            st.write("🎬 Recommended Movies:", movies)
        else:
            st.warning("No movies found for this emotion in dataset.")
