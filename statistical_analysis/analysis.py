# importing necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from mtcnn import MTCNN
import argparse


# loading models with error handling
def load_models(gender_model_path, age_model_path):
    try:
        gender_model = tf.keras.models.load_model(gender_model_path)
        age_model = tf.keras.models.load_model(age_model_path)
        print('Models loaded successfully!')
        return gender_model, age_model
    except FileNotFoundError:
        print("Error: Model file not found. Please check the file paths.")
        return None, None
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None


# crops images from a given directory and saves them to an output directory
def process_and_crop_mtcnn(input_dir, output_dir):

    # error handling for directories
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")

    detector = MTCNN()

    # iterating over images in directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        try:
            img = Image.open(img_path)
            img_array = np.array(img)

            # converts image to RGB if it has an alpha channel
            if img_array.shape[-1] == 4:
                img = img.convert("RGB")
                img_array = np.array(img)

            detections = detector.detect_faces(img_array)

            if detections:
                for i, face in enumerate(detections):
                    x, y, width, height = face['box']
                    x2, y2 = x + width, y + height

                    # cropping and resizing face
                    face_img = img_array[y:y2, x:x2]

                    face_img_pil = Image.fromarray(face_img)
                    face_img_resized = face_img_pil.resize((224, 224))

                    # saving cropped and resized image to output directory
                    output_img_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_face_{i+1}.jpg")
                    face_img_resized.save(output_img_path)

        except Exception as e:
            print(f"Error processing {img_name}: {e}")


# saves the summary of the metrics used along with plots
def generate_summary_with_plots(df_results, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Output directory '{save_dir}' created.")

   # gender count and male to female ratio
    male_count = df_results[df_results['Gender'] == "Male"].shape[0]
    female_count = df_results[df_results['Gender'] == "Female"].shape[0]

    if female_count > 0:
        male_female_ratio = male_count / female_count
        print(f"\nMale to Female Ratio: {male_female_ratio:.2f}")
    else:
        print("\nNo females detected; ratio calculation is not possible.")

    # gender confidence metrics
    avg_male_confidence = (df_results[df_results['Gender'] == "Male"]["Gender Confidence"].mean())*100
    avg_female_confidence = (df_results[df_results['Gender'] == "Female"]["Gender Confidence"].mean())*100

    # age distribution metrics
    mean_age = df_results['Age'].mean()
    median_age = df_results['Age'].median()
    mode_age = df_results['Age'].mode().values[0]
    age_range = (df_results['Age'].min(), df_results['Age'].max())
    std_dev_age = df_results['Age'].std()

    # creating a summary DataFrame
    summary = pd.DataFrame({
    "Metric": ["Total Faces", "Number of Males", "Number of Females", "Male to Female Ratio",
               "Average Male Confidence", "Average Female Confidence",
               "Mean Age", "Median Age", "Mode Age", "Age Range", "Age Standard Deviation"],
    "Value": [male_count + female_count, male_count, female_count, f"{male_female_ratio:.2f}",
              f"{avg_male_confidence:.2f}%", f"{avg_female_confidence:.2f}%",
              mean_age, median_age, mode_age, f"{age_range[0]} - {age_range[1]}", std_dev_age]})

    # saving the DataFrame as an Excel file in the given directory
    summary.to_excel(os.path.join(save_dir, 'summary.xlsx'), index=False)

    # data plots
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Gender", y="Age", data=df_results)
    plt.title("Age Distribution by Gender")
    plt.savefig(os.path.join(save_dir,"age_distribution_by_gender.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(df_results["Age"], kde=True, binwidth=1)
    plt.title("Age Distribution")
    plt.savefig(os.path.join(save_dir,"age_distribution.png"))
    plt.xlabel("Age")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_results["Gender"])
    plt.title("Gender Count")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir,"gender_count.png"))
    plt.close()

    return summary


# main function to analyze faces
def analyze_faces(directory, gender_model, age_model, output_directory, summary_and_plot_directory):    
    process_and_crop_mtcnn(directory, output_directory)
    results = []

    for img_name in os.listdir(output_directory):
      img_path = os.path.join(output_directory, img_name)

      try:
          img = Image.open(img_path)
          img_array = np.array(img) / 255.0
          img_array = np.expand_dims(img_array, axis=0)

          gender_pred_prob = gender_model.predict(img_array)
          gender_pred = (gender_pred_prob[0][0] > 0.5).astype(int)
          gender_confidence = gender_pred_prob[0][0] if gender_pred == 1 else 1 - gender_pred_prob[0][0]

          gender_label = "Male" if gender_pred == 1 else "Female"

          age_pred = age_model.predict(img_array)
          age = int(age_pred[0][0])

          results.append({
              "Image": img_name,
              "Gender": gender_label,
              "Gender Confidence": gender_confidence,
              "Age": age
          })

      except Exception as e:
          print(f"Error analyzing {img_name}: {e}")

    # convert results to DataFrame
    df_results = pd.DataFrame(results)

    # saving the DataFrame as an Excel file
    df_results.to_excel(os.path.join(output_directory, 'results.xlsx'), index=False)

    summary = generate_summary_with_plots(df_results, summary_and_plot_directory)

    return summary


# specify age and gender model paths here
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze faces in images and provide demographic insights.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Path to the directory where cropped images will be saved.")
    parser.add_argument("--summary_dir", type=str, default="summary_and_plots", help="Path to the directory where summary and plots will be saved.")
    parser.add_argument("--gender_model", type=str, default="../models/gender_model.keras", help="Path to the gender model file.")
    parser.add_argument("--age_model", type=str, default="../models/age_model.keras", help="Path to the age model file.")
    args = parser.parse_args()

    gender_model, age_model = load_models(args.gender_model, args.age_model)
    if gender_model and age_model:
        analyze_faces(args.input_dir, gender_model, age_model, args.output_dir, args.summary_dir)