import os
from PIL import Image
import torch
import clip
import numpy as np
import pickle
import streamlit as st
from torchvision import models, transforms
import zipfile
import requests
import gdown
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Interior AI Search", layout="wide")
device = "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

def download_model_from_gdrive():
    url = "https://drive.google.com/uc?export=download&id=1OvbfsDQhnxd72-CZUjIGcop7dJzBHWMu"
    output = "style_classifier_stage3.pth"
    gdown.download(url, output, quiet=False)

def download_and_extract_thumbnails():
    url = "https://drive.google.com/uc?export=download&id=1ljPKI67c50gpu2X5NySQVqHqsxH9Pbjk"
    zip_path = "thumbnails.zip"
    if not os.path.exists("thumbnails"):
        print("Downloading thumbnails.zip...")
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Extracting thumbnails...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("thumbnails")
        print("Thumbnails ready.")

download_and_extract_thumbnails()

# Load the ResNet-based style classifier
@st.cache_resource
def load_classifier():
    if not os.path.exists("style_classifier_stage3.pth"):
        download_model_from_gdrive()
    model_resnet = models.resnet50(pretrained=False)
    model_resnet.fc = torch.nn.Linear(model_resnet.fc.in_features, 19)
    model_resnet.load_state_dict(torch.load("style_classifier_stage3.pth", map_location=device))
    model_resnet.eval()
    return model_resnet

# Load CLIP data (features, paths) for retrieval
@st.cache_resource
def load_data():
    with open("clip_data.pkl", "rb") as f:
        data = pickle.load(f)
    features = data["features"]
    paths = data["paths"]

    return model, preprocess, features, paths

model, preprocess, features, image_paths = load_data()
classifier = load_classifier()

thumbnail_dir = "thumbnails"

# ----------------------------
# Functions
# ----------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_labels = [
    'rustic', 'asian', 'modern', 'french-country', 'southwestern',
    'coastal', 'scandinavian', 'eclectic', 'traditional', 'transitional',
    'craftsman', 'mediterranean', 'farmhouse', 'contemporary', 'tropical',
    'mid-century-modern', 'industrial', 'victorian', 'shabby-chic-style'
]

def predict_style(image, model):
    model = model.to(device)
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        _, pred_idx = torch.max(probs, dim=1)
    predicted_style = class_labels[pred_idx.item()]
    return predicted_style

def search_by_text(prompt, top_k=5):
    with torch.no_grad():
        text = clip.tokenize([prompt]).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()

    similarities = np.array([1 - cosine(text_features, feature) for feature in features])
    indices = np.argsort(similarities)[::-1][:top_k]
    return [image_paths[i] for i in indices]

def search_by_image(uploaded_file, top_k=5):
    image = Image.open(uploaded_file).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()

    similarities = np.array([1 - cosine(image_features, feature) for feature in features])
    indices = np.argsort(similarities)[::-1][:top_k]
    return [image_paths[i] for i in indices]

from transformers import BlipProcessor, BlipForConditionalGeneration
@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model
processor_blip, model_blip = load_caption_model()

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor_blip(raw_image, return_tensors="pt").to(device)
    out = model_blip.generate(**inputs)
    caption = processor_blip.decode(out[0], skip_special_tokens=True)
    return caption

def boost_style_matches(retrieved_paths, predicted_style):
    # No boost logic needed if not displaying scores — keeping structure for future use
    return retrieved_paths

import math

def show_image_grid(paths, columns=3):
    if not paths:
        st.warning("No images to display.")
        return

    rows = math.ceil(len(paths) / columns)
    for i in range(rows):
        cols = st.columns(columns)
        for j in range(columns):
            idx = i * columns + j
            if idx < len(paths):
                with cols[j]:
                    st.image(paths[idx], use_column_width=True)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🧠 AI-Powered Interior Design Image Search")

tab1, tab2 = st.tabs(["🔍 Text Search", "🖼️ Image Upload"])

# Text Search Tab
with tab1:
    query = st.text_input("Describe your ideal room style:")
    selected_filter = st.selectbox("Filter by style (optional)", ["All"] + class_labels)
    if query:
        st.info("Searching for matches...")
        results = search_by_text(query)
        # If filter selected, narrow down
        if selected_filter != "All":
            results = [r for r in results if selected_filter in r]  # Assumes path includes label

        show_image_grid(results)

# Image Upload Tab
with tab2:
    uploaded_file = st.file_uploader("Upload a reference image", type=["jpg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded image", use_column_width=False)

        # Predict style
        image = Image.open(uploaded_file).convert("RGB")
        predicted_style = predict_style(image, classifier)
        st.markdown(f"### 🎨 Predicted Style: `{predicted_style}`")

        # Retrieve and display
        results = search_by_image(uploaded_file)
        boosted_results = boost_style_matches(results, predicted_style)
        show_image_grid(boosted_results)

st.markdown("---")
st.caption("Powered by OpenAI CLIP and Streamlit ❤️")
