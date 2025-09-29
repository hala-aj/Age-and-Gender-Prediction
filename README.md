
# Age and Gender Prediction

This deep learning project leverages facial recognition technology to detect faces in images, providing age and gender predictions for each detected face. Statistical analyses are generated automatically based on these detections and predictions, offering insight into demographics in a real-world scenario.


## Table of Contents
 
 - [Demo](#demo)
 
 - [Dataset](#dataset)

 - [Project structure](#project-structure)

 - [Run Locally](#run-locally)
 
 - [Contributors](#contributors)

 - [License](#license)
## Demo
This is the streamlit interface 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()

<p align="center">
<img src="https://github.com/hala-aj/Age-and-Gender-Prediction/blob/main/media/streamlit_1.png?raw=true" alt="demoimage1" /><br>
<img src="https://github.com/hala-aj/Age-and-Gender-Prediction/blob/main/media/streamlit_2.png?raw=true" alt="demoimage2" />
</p>


You can access it through this [link]() or set it up locally by following the [installation steps](#run-locally) provided.
## Dataset
The [IMDB-Clean](https://www.kaggle.com/datasets/yuulind/imdb-clean) dataset was utilized due to its diverse range of annotated facial images. This dataset, containing over 200k images, is suitable for training deep-learning models due to its structured format and comprehensive labeling.
<p align="center">
<img src="https://github.com/hala-aj/Age-and-Gender-Prediction/blob/main/media/dataset-cover.png?raw=true" alt="dataset" width="100%" />
</p>

## Project Structure

    ├──age_training.ipynb
    ├──gender_training.ipynb
    ├──predictions.py
    ├──predictions_streamlit.py
    ├──statistical_analysis
      ├──analysis.ipynb
      └──analysis.py
    ├──requirements.txt
    └──README.md
       
1. [age_training.ipynb]() & [gender_training.ipynb](): Training models on the datasets
- utilized transfer learning by using the VGG16 pre-trained model
- used proper augmentation and data preprocessing
- implemented multiple callbacks to monitor performance
- unfroze some layers to aid in assigning weights

2. [predictions.py](): Loads trained models to make predictions on detected faces

- utilizes MTCNN (Multi-Task Cascaded Convolutional Neural Networks) for face detection 
- uses PIL (Python Imaging Library) to draw color-coded boxes over detected faces
- passes cropped images to the trained models to generate prediction labels
- generates different label styles depending on the face proximity (to prevent label overlap)
- displays a table containing the faces detected with their labels and confidence percentages

3. [predictions_streamlit.py](): Streamlit app version of [predictions.py]() that loads models directly from Hugging Face instead of local paths

4. [analysis.ipynb](): provides statistical analysis of a given directory and saves the results in a specified directory
- enables user interactivity by allowing users to modify metrics applied to the dataset they provide, enhancing the analytical experience and tailoring insights to specific needs
- utilizes MTCNN to crop the faces in the input directory and save them to the specified output directory along with their predictions in csv format
- saves a summary of the data to a csv file along with plots

5. [analysis.py](): compact form of [analysis.ipynb]() to run locally (for automation)

6. [requirements.txt](): file containing all the necessary dependencies for environment setup



### Models
This repository does not include the trained model files. Download them from:  https://huggingface.co/halaaj/AgeGenderModels

Place these files in a `models/` folder at the repository root:

- models/age_model.keras
- models/gender_model.keras

The analysis and predictions scripts assume a default model directory. Moving the models will cause errors, so either keep them in place or specify both model paths.

## Run Locally

First ensure the following software is installed.

- Python: version 3.12
- Git: for cloning the repository
- Pip: Python's package installer (confirm with this)
```bash
  python --version
  git --version
  pip --version
```
Clone the repository

```bash
  git clone https://github.com/hala-aj/Age-and-Gender-Prediction
```

Go to the project directory

```bash
  cd Age-and-Gender-Prediction
```
Create a virtual environment to manage dependencies

```bash
  python -m venv .venv
```

Activate the virtual environment *(confirm by seeing **.venv** in command prompt)*

- Windows:
```bash
  .venv\Scripts\activate
```
- Mac/Linux:
```bash
  source .venv/bin/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```
Run the Streamlit Application
```bash
  streamlit run predictions.py
```

Run the Analysis script *(for statistical analysis)*

```bash
  python analysis.py --input_dir /path/to/input  
```
*Can also be run with more arguments like:* 

```bash
  python analysis.py --input_dir /path/to/input  --output_dir /path/to/output
  python analysis.py --help
```
 *(--help/-h will give the list of arguments the script can take)*
  #### Ensure the proper project structure (including models) is set up before running the scripts.




## Contributors
*This project was developed by:*

**Mazen Alobeid** (click [here](https://github.com/Mazen-9) for github profile)
    
[![linkedin](https://img.shields.io/badge/linkedin-b6d7a8?style=for-the-badge&logo=linkedin&logoColor=black)](https://www.linkedin.com/in/mazen-alobeid/)

**Hala Alajrad** (click [here](https://github.com/hala-aj) for github profile)
    
[![linkedin](https://img.shields.io/badge/linkedin-a2c4c9?style=for-the-badge&logo=linkedin&logoColor=black)](https://www.linkedin.com/in/hala-alajrad/)


## License

[MIT](https://github.com/hala-aj/Age-and-Gender-Prediction?tab=MIT-1-ov-file)
