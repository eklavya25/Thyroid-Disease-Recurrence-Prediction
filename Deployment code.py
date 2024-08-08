import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

# Read the dataset
df = pd.read_csv("Thyroid_Diff_2.csv")

# Encode categorical variables
le_dict = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        le_dict[column] = le

# Save the LabelEncoders
joblib.dump(le_dict, 'label_encoders.joblib')

# Split the data into features and labels
X = df.drop('Recurred', axis=1)
y = df['Recurred']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Hyperparameters for GridSearchCV
rf_params = {'classifier__n_estimators': [10, 50, 100, 200], 'classifier__max_depth': [None, 10, 20, 30]}

# Train the Random Forest model
grid_rf = GridSearchCV(pipeline_rf, rf_params, cv=5)
grid_rf.fit(X_train, y_train)

# Save the trained model
joblib.dump(grid_rf, 'rf_model.joblib')



import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import joblib
import numpy as np

# Load the trained model and LabelEncoders
model = joblib.load('rf_model.joblib')
le_dict = joblib.load('label_encoders.joblib')

# Streamlit app setup
st.title("Thyroid Disease Prediction")

# Input form
st.header("Patient Details")
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=['M', 'F'])
smoking = st.selectbox("Smoking", options=['Yes', 'No'])
hx_smoking = st.selectbox("Hx Smoking", options=['Yes', 'No'])
hx_radiotherapy = st.selectbox("Hx Radiothreapy", options=['Yes', 'No'])
thyroid_function = st.selectbox("Thyroid Function", options=['Euthyroid', 'Clinical Hyperthyroidism', 'Clinical Hypothyroidism',
       'Subclinical Hyperthyroidism', 'Subclinical Hypothyroidism'])
physical_exam = st.selectbox("Physical Examination", options=['Single nodular goiter-left', 'Multinodular goiter',
       'Single nodular goiter-right', 'Normal', 'Diffuse goiter'])
adenopathy = st.selectbox("Adenopathy", options=['No', 'Right', 'Extensive', 'Left', 'Bilateral', 'Posterior'])
pathology = st.selectbox("Pathology", options=['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'])
focality = st.selectbox("Focality", options=['Uni-Focal', 'Multi-Focal'])
risk = st.selectbox("Risk", options=['Low', 'Intermediate', 'High'])
T = st.selectbox("T", options=['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'])
N = st.selectbox("N", options=['N0', 'N1b', 'N1a'])
M = st.selectbox("M", options=['M0', 'M1'])
stage = st.selectbox("Stage", options=['I', 'II', 'IVB', 'III', 'IVA'])
response = st.selectbox("Response", options=['Indeterminate', 'Excellent', 'Structural Incomplete',
       'Biochemical Incomplete'])

# Create a dictionary for the input features
input_data = {
    'Age': age,
    'Gender': gender,
    'Smoking': smoking,
    'Hx Smoking': hx_smoking,
    'Hx Radiothreapy': hx_radiotherapy,
    'Thyroid Function': thyroid_function,
    'Physical Examination': physical_exam,
    'Adenopathy': adenopathy,
    'Pathology': pathology,
    'Focality': focality,
    'Risk': risk,
    'T': T,
    'N': N,
    'M': M,
    'Stage': stage,
    'Response': response
}

# Convert categorical variables using the loaded LabelEncoders
for column in input_data.keys():
    if column in le_dict:
        input_data[column] = le_dict[column].transform([input_data[column]])[0]

# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make the prediction
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_df)[0][1]
    st.write(f"The probability of cancer recurrence is: {prediction_proba:.2f}")
