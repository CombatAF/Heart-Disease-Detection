# Heart-Disease-Detection
Uses various medical paramaters to detect heart disease

Heart Disease Detection Using Machine Learning
Overview
This project aims to predict the likelihood of heart disease in patients based on various health metrics. By leveraging machine learning algorithms, the system analyzes medical data to provide early insights, helping in timely medical intervention.

Features
Data Preprocessing: Handles missing values, encodes categorical data, and scales numerical features.
Exploratory Data Analysis (EDA): Provides insights into the data through statistical analysis and visualizations.
Model Training and Evaluation:
Trains machine learning models to classify patients with or without heart disease.
Evaluates models using metrics like accuracy, precision, recall, and F1-score.
Prediction System: Accepts patient data as input and predicts the likelihood of heart disease.
Technologies Used
Programming Language: Python
Libraries:
numpy for numerical operations.
pandas for data manipulation.
matplotlib and seaborn for visualizations.
scikit-learn for machine learning models and evaluation.
Streamlit for building an interactive web app.
Dataset
The dataset contains the following features:

Age: Age of the patient.
Sex: Gender of the patient (Male: 1, Female: 0).
Chest Pain Type: Type of chest pain (0–3 categories).
Resting Blood Pressure: Measured in mmHg.
Cholesterol: Serum cholesterol in mg/dl.
Fasting Blood Sugar: >120 mg/dl (1: True, 0: False).
Resting ECG: Electrocardiographic results (0–2 categories).
Maximum Heart Rate Achieved
Exercise-Induced Angina: (1: Yes, 0: No).
ST Depression: Induced by exercise relative to rest.
Thalassemia:
0: Unknown
1: Normal
2: Fixed Defect
3: Reversible Defect
Target: Presence (1) or absence (0) of heart disease.
You can use datasets like the UCI Heart Disease dataset.

Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/heart-disease-detection.git
cd heart-disease-detection
Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Usage
Load Dataset: Reads data from heart_disease.csv.
Preprocess Data: Encodes categorical variables and scales numerical features.
Train Model: Trains models like Logistic Regression, Random Forest, or Support Vector Machines.
Evaluate Model: Validates model performance using test data.
Predict: Provides predictions for new patient data via the web interface.
Results
Training Accuracy: ~90%
Test Accuracy: ~88%
Precision, Recall, and F1-score: Evaluated for balanced predictions.
Example
Input Example (Streamlit app):

Age: 55
Sex: Male
Chest Pain Type: Typical Angina
Resting BP: 140
Cholesterol: 200
Fasting Blood Sugar: Yes
Thalassemia: Reversible Defect
Prediction:

css
Copy code
The model predicts a 75% likelihood of heart disease.
Future Improvements
Incorporate additional features like stress test results or imaging data.
Improve the model with deep learning algorithms.
Add a feature for risk analysis with medical recommendations.
Deploy on a cloud platform for real-time predictions.
