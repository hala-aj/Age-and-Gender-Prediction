# importing necessary libraries
import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from mtcnn import MTCNN
import streamlit as st
import pandas as pd
import imghdr
from pathlib import Path
from huggingface_hub import hf_hub_download

# HuggingFace Configuration
HF_REPO = "halaaj/AgeGenderModels"
AGE_FILENAME = "age_model.keras"
GENDER_FILENAME = "gender_model.keras"


# cache directory for downloaded models
CACHE_DIR = Path.home() / ".cache" / "age_gender_models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# downloads file from huggingface (if not cached) and returns local path
def get_model_path(repo_id: str, filename: str) -> str:
    # local cache path
    local_path = CACHE_DIR / filename
    if local_path.exists():
        return str(local_path)
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(CACHE_DIR))
        return str(downloaded)
    except Exception as e:
        raise RuntimeError(f"Failed to download {filename} from {repo_id}: {e}")

# loading pretrained models
gender_model = None
age_model = None

def download_and_load_models():
    global gender_model, age_model
    if gender_model is not None and age_model is not None:
        return gender_model, age_model

    try:
        gender_model_path = get_model_path(HF_REPO, GENDER_FILENAME)
        age_model_path = get_model_path(HF_REPO, AGE_FILENAME)
        gender_model = tf.keras.models.load_model(gender_model_path)
        age_model = tf.keras.models.load_model(age_model_path)
        print("Models loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Error loading models: {e}")
    return gender_model, age_model

# validating MIME type and converting image to RGB
def validate_image(uploaded_file):
    try:
        mime_type = imghdr.what(uploaded_file)
        if mime_type not in ['jpeg', 'png', 'jpg']:
            return None

        img = Image.open(uploaded_file)
        img.verify()
        img = Image.open(uploaded_file)
        return img.convert('RGB')
    except (IOError, SyntaxError):
        return None

# calculates distance between two bounding boxes
def calculate_distance(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    # calculate center points of both boxes
    center_x1, center_y1 = x1 + width1 / 2, y1 + height1 / 2
    center_x2, center_y2 = x2 + width2 / 2, y2 + height2 / 2

    # Euclidean distance between the centers of the boxes
    distance = math.sqrt((center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2)
    return distance


# returns the faces detected with their predictions
def analyze_faces(img):
    # ensure models are loaded
    g_model, a_model = download_and_load_models()

    detector = MTCNN()
    img_array = np.array(img)

    detections = detector.detect_faces(img_array)
    predictions = []

    if detections:
        face_imgs = []
        for face in detections:
            x, y, width, height = face['box']
            x2, y2 = x + width, y + height

            # storing extracted faces into an array
            face_img = img_array[y:y2, x:x2]
            face_img_resized = Image.fromarray(face_img).resize((224, 224))
            img_array_resized = np.array(face_img_resized) / 255.0
            img_array_resized = np.expand_dims(img_array_resized, axis=0)
            face_imgs.append(img_array_resized)

        # batch predictions for efficiency
        face_imgs = np.vstack(face_imgs)

        age_preds = a_model.predict(face_imgs)
        gender_preds_prob = g_model.predict(face_imgs)

        for i in range(len(detections)):
            age_label = int(age_preds[i][0])
            gender_pred = (gender_preds_prob[i][0] > 0.5).astype(int)
            gender_label = "Male" if gender_pred == 1 else "Female"
            gender_confidence = gender_preds_prob[i][0] if gender_pred == 1 else 1 - gender_preds_prob[i][0]

            predictions.append({
                "Face": f"Face {i+1}",
                "Gender": gender_label,
                "Confidence": f"{gender_confidence * 100:.2f}%",
                "Age": age_label
            })

    return detections, predictions


# outlines text to appear properly on image
def draw_outlined_text(draw, position, text, font, text_color, outline_color, outline_width=2):
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
    draw.text((x, y), text, font=font, fill=text_color)


# returns image with boxes drawn over the faces with their label style depending on proximity
def draw_boxes_with_proximity(image, detections, predictions, proximity_threshold=100):
    draw = ImageDraw.Draw(image)
    try:
        large_font = ImageFont.truetype("arial.ttf", 28)
        small_font = ImageFont.truetype("arial.ttf", 13)
    except IOError:
        large_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    num_faces = len(detections)

    for i, (detection, prediction) in enumerate(zip(detections, predictions)):
        x, y, width, height = detection['box']
        x2, y2 = x + width, y + height

        face_label = prediction["Face"]
        gender_label = prediction["Gender"]
        age_label = prediction["Age"]

        # checks when face distances are below the threshold
        is_close = False
        for j in range(num_faces):
            if i != j:
                distance = calculate_distance(detection['box'], detections[j]['box'])
                if distance < proximity_threshold:
                    is_close = True
                    break

        # if faces are close, only draw outlined text
        if is_close:
            # text describing face number
            outlined_text_position = (x, y - 20)
            draw_outlined_text(draw, outlined_text_position, face_label, small_font, text_color="#FFFFFF", outline_color="#5b5b5b", outline_width=2)

            # prediction label
            label = f"{gender_label} {age_label}"
            outlined_text_position_label = (x, y2 + 5)  # Adjusted position below the bounding box
            draw_outlined_text(draw, outlined_text_position_label, label, small_font, text_color="#FFFFFF", outline_color="#5b5b5b", outline_width=1)

            # color coded face rectangle 
            outline_color = "#15355a" if gender_label == "Male" else "#80004b"
            draw.rectangle([x, y, x2, y2], outline=outline_color, width=2)

        else:
            # text describing face number
            outlined_text_position = (x, y - 30)
            draw_outlined_text(draw, outlined_text_position, face_label, large_font, text_color="#FFFFFF", outline_color="#5b5b5b", outline_width=2)

            # color coded face rectangle
            outline_color = "#15355a" if gender_label == "Male" else "#80004b"
            draw.rectangle([x, y, x2, y2], outline=outline_color, width=2)

            # prediction label
            label = f"{gender_label} {age_label}"

            # placing the prediction label below the bounding box
            text_bbox = draw.textbbox((x, y2 + 10), label, font=large_font)
            padding = 10

            # drawing the outlined rectangle
            text_background_x1 = text_bbox[0] - padding // 2
            text_background_y1 = text_bbox[1] - padding // 2
            text_background_x2 = text_bbox[2] + padding // 2
            text_background_y2 = text_bbox[3] + padding // 2

            draw.rectangle([text_background_x1 - 2, text_background_y1 - 2, text_background_x2 + 2, text_background_y2 + 2], fill="#999999")
            draw.rectangle([text_background_x1, text_background_y1, text_background_x2, text_background_y2], fill="#055351")
            draw.text((text_background_x1 + padding // 2, text_background_y1), label, fill="white", font=large_font)

    return image



def main():
    st.title("Age and Gender Prediction")
    st.write("Upload an image. Faces will be detected and the predictions will be displayed.")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:

        image = validate_image(uploaded_file)
        if image is None:
            st.error("Invalid image file. Please upload a valid JPEG or PNG image.")
            return

        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze"):
            with st.spinner('Processing...'):
                try:
                    download_and_load_models()
                    detections, predictions = analyze_faces(image)
                except Exception as e:
                    st.error(f"Model error: {e}")
                    return

            if detections:
                image_with_boxes = draw_boxes_with_proximity(image.copy(), detections, predictions)
                st.image(image_with_boxes, caption='Processed Image', use_column_width=True)

                df = pd.DataFrame(predictions).set_index("Face")
                st.table(df)

            else:
                st.write("No faces detected in the image.")

if __name__ == '__main__':
    main()
