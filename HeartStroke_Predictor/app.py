import streamlit as st
import pandas as pd
import joblib

model=joblib.load('KNN_model.pkl')
scaler=joblib.load('scaler.pkl')
expected_columns=joblib.load('columns.pkl')

st.title("Heart Stroke Prediction Model❤️")
st.markdown("This app predicts the likelihood of a heart stroke based on user input features. Please fill in the required information below and click 'Predict' to see the results.")
age=st.slider("Age",18,100,40)
sex=st.selectbox('Sex',['Male','Female'])
chest_pain_type=st.selectbox('Chest Pain Type',["ATA","NAP","ASY","TA"])
resting_BP=st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol=st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_BS=st.selectbox('Fasting Blood Sugar > 120 mg/dL',[1,0])
resting_ECG=st.selectbox('Resting ECG Results',["Normal","ST","LVH"])
max_HR=st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina=st.selectbox('Exercise Induced Angina',['Y','N'])
oldpeak=st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 10.0, 1.0)
st_slope=st.selectbox('Slope of the Peak Exercise ST Segment',["Up","Flat","Down"])

if st.button("Predict"):
    raw_input={
        "Age": age,
        "Sex_"+sex: 1,
        "ChestPainType_"+chest_pain_type: 1,
        "RestingBP": resting_BP,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_BS,
        "RestingECG_"+resting_ECG: 1,
        "MaxHR": max_HR,
        "ExerciseAngina_"+exercise_angina: 1,
        "Oldpeak": oldpeak,
        "ST_Slope_"+st_slope: 1
    }
    
    input_df=pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df=input_df[expected_columns]
    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)[0]
    
    if prediction==1:
        st.error("The model predicts that you are at risk of a heart stroke. Please consult a healthcare professional for further evaluation.")
    else:
        st.success('''Congratulations!❤️. You are Safe. 
                   The model predicts that you are not at risk of a heart stroke. 
                   However, please maintain a healthy lifestyle and consult a healthcare professional for regular check-ups.''') 
        