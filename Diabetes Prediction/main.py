import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('Model.pkl', 'rb'))

def run():
    st.title('Diabetes Prediction Web App')

    pregnancies = st.number_input("Pregnancies", value = 0)

    glucose = st.number_input("Glucose", value = 0.0, format = '%.1f')

    blood_pressure = st.number_input('Blood Pressure', value = 0.0, format = '%.1f')

    skin_thickness = st.number_input('Skin Thickness', value = 0.0, format = '%.1f')

    Insulin = st.number_input('Insulin', value = 0.0, format = '%.1f')

    BMI = st.number_input('BMI', value = 0.0, format = '%.1f')

    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Funstion', value = 0.0, format = '%.1f')

    age = st.number_input('Age', value = 0)

    data = [[pregnancies, glucose, blood_pressure, skin_thickness, Insulin, BMI, diabetes_pedigree_function, age]]

    df = pd.DataFrame(data)

    sc = StandardScaler()
    scaled_data = sc.fit_transform(data)

    if st.button('Predict'):

        prediction = model.predict(scaled_data)

        if prediction == 0:
            st.success("You Don't Have Diabetes")

        else:
            st.error("You Have Diabetes, Please contact to a doctor")


run()
