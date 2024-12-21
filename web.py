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
st.title("气道播散")
# Age: numerical input
Percentage = st.number_input ("Percentage of solid components:", min_value=0.0, max_value=1.0, value=0.5)
# Sex: categorical selection
# Chest Pain Type (ср): categorical selection
stages = st.selectbox("stages:", options=[0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13])
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
        advice = (f"根据模型预测，您患有心脏疾病的风险较高。")  
    else:  
        advice = (f"根据模型预测，您患有心脏疾病的风险较低。")  
    
    st.write(advice)  
