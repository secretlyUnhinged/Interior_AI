# AI-Powered Interior Design Image Search

This project is an AI-powered interior design image search application that leverages OpenAI's CLIP model and other machine learning models to assist users in searching for interior design images based on text descriptions or uploaded reference images. The app also allows users to predict room styles from uploaded images.

## Features

- Text Search: Search for interior design images by describing your ideal room style.
- Image Upload: Upload a reference image to find similar design images and predict the room style.
- Style Prediction: Predict the style of a room based on an uploaded image (e.g., rustic, modern, etc.).
- Image Retrieval: Use CLIP and FAISS for efficient image retrieval based on the input description or image.
- Room Style Classifier: A ResNet-based classifier predicts room styles from the uploaded images.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/secretlyUnhinged/Interior_AI.git
cd Interior_AI
````

### 2. Set up dependencies

Create a virtual environment and install the required dependencies.

```bash
# Create virtual environment
python3 -m venv env
# Activate the virtual environment
source env/bin/activate  # For Mac/Linux
env\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Download the required files

The application requires the following files:

* **Style Classifier Model** (`style_classifier_stage3.pth`)
* **CLIP Data** (`clip_data.pkl`)
* **Thumbnails ZIP** (`thumbnails.zip`)

#### 3.1 Thumbnails

The thumbnail images are not included in the GitHub repository due to their large size. To use the app, you will need to download the `thumbnails.zip` file from Google Drive.

* [Download Thumbnails ZIP](https://drive.google.com/uc?export=download&id=%%%)

Once downloaded, extract the `thumbnails.zip` into the `thumbnails` directory of the project.

#### 3.2 Model Files

You should already have the model weights for the style classifier (`style_classifier_stage3.pth`) and CLIP data (`clip_data.pkl`). If you don't have these, you can obtain them from the relevant sources or request them from the project maintainers.

### 4. Run the app

You can now run the app using Streamlit.

```bash
streamlit run app.py
```

This will start a local development server, and you can access the app at `http://localhost:8501` in your browser.

## Project Structure

```bash
Interior_AI/
│
├── app.py                    # Main Streamlit app script
├── style_classifier_stage3.pth  # Pretrained room style classifier model
├── clip_data.pkl             # CLIP model data for image retrieval
├── thumbnails.zip            # Compressed thumbnails (must be downloaded separately)
├── thumbnails/               # Folder for extracted thumbnails
├── requirements.txt          # List of Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore file to exclude unwanted files
```

## How It Works

### Image Search

The application uses OpenAI's CLIP model to search for interior design images based on either a text prompt or an uploaded reference image. Here's how it works:

* Text Search: The app tokenizes the text prompt using CLIP's tokenizer, then computes text features, and searches for the most similar images in the dataset using FAISS (Fast Approximate Nearest Neighbors).
* Image Search: The app extracts features from the uploaded reference image using CLIP and searches for similar images in the dataset.

### Style Prediction

The app also includes a room style classifier based on a pretrained ResNet-50 model. The model predicts the style of an uploaded reference image (e.g., modern, rustic, industrial). The app uses this style prediction to narrow down image search results that match the predicted style.

### Thumbnails

The app uses thumbnails for efficient display of search results. These images are downloaded and extracted from a ZIP file hosted on Google Drive.

## Contribution

Contributions are welcome! If you have suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```

Just copy this entire block of text and paste it directly into your `README.md` file on GitHub. Make sure to replace `YOUR_GOOGLE_DRIVE_FILE_ID` with the actual file ID for the thumbnails zip hosted on your Google Drive.
```
