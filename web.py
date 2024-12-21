import streamlit as st 
import joblib
import numpy as np 
import pandas as pa 
import shap
import pandas as pd
# Load the new model
model = joblib.load ('LightGBM.pkl')
# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('clinic_fill.csv')
# Define feature names from the new dataset
feature_names =['Percentage of solid components',	'stages',	'boundary',	'lobulation',	'CEA'	,'SCC'	,'ith_score'	,'ith_sign']	


# Streamlit user interface
st.title("Prediction model of airway spread")
# Age: numerical input
Percentage = st.number_input ("Percentage of solid components:", min_value=0.0, max_value=1.0, value=0.5)
# Sex: categorical selection
# Chest Pain Type (ср): categorical selection

options_stage = [  
    "0 - T2N0M0",  
    "1 - T2N1M0",  
    "2 - T2N2M0",  
    "3 - T1N0M0",  
    "4 - T1N2M0",  
    "5 - T3N2M0",  
    "6 - T1N1M0",  
    "7 - T3N0M0",  
    "8 - T2N0M1",  
    "9 - T1N0M0",  
    "10 - T1N0M1",  
    "11 - T1N2M0",  
    "12 - T2N0M0",  
    "13 - T3N0M0",  
]  

stages = st.selectbox("stages:", options=options_stage)  
boundary = st.selectbox("boundary:", options=[0, 1])
lobulation = st.selectbox("lobulation:", options=[0, 1])
CEA_value = st.number_input("CEA:",  min_value=0, max_value=10000, value=5)
SCC_value = st.number_input("SCC:",  min_value=0, max_value=10000, value=5)
ith_score = st.number_input ("ith_score:", min_value=0.0, max_value=1.0, value=0.5)
ith_sign = st.number_input ("ith_sign:", min_value=0.0, max_value=1.0, value=0.5)

feature_values = [Percentage, stages, boundary, lobulation, CEA_value, SCC_value, ith_score, ith_sign]
features=np.array(feature_values)

if st.button("Predict"):  
    # 将 feature 数组从一维转为二维  
    features = features.reshape(1, -1)  

    # 预测类别和分类概率  
    predicted_class = model.predict(features)[0]  
    predicted_proba = model.predict_proba(features)[0]  

    # 根据预测结果生成建议  
    probability = predicted_proba[predicted_class] * 100  

    if predicted_class == 1:  
        advice = (f"According to the model prediction, the risk of tumor airway dispersion is high.")  
    else:  
        advice = (f"According to the model prediction, the risk of tumor airway dispersal is low.")  
    
    st.write(advice)  
