import streamlit as st
import joblib
import numpy as np

model = joblib.load('lr.pkl')

def predict_admission(gre, toefl, universityrating, sop, lor, cgpa, research):
    features = np.array([gre, toefl, universityrating, sop, lor, cgpa, research]).reshape(1,-1)
    admission_chance = model.predict(features)[0]
    return admission_chance

def map_to_percentage(value):
    if value > 1:
        value = 1
    return value * 100

def main():
    st.title(':red[Graduate Admission Predictor]')
    st.write(':blue[Enter your GRE, TOEFL, University Rating, Statement of Purpose, Letter of Recommendation, CGPA, and Research Experience to find out your chance of admission!]')
    gre = st.slider('GRE Score', 0, 340, 300)
    toefl = st.slider('TOEFL Score', 0, 120, 100)
    universityrating = st.slider('University Rating', 0, 5, 3)
    sop = st.slider('Statement of Purpose', 0.0, 5.0, 3.0, 0.1)
    lor = st.slider('Letter of Recommendation', 0.0, 5.0, 3.0, 0.1)
    cgpa = st.slider('CGPA', 0.0, 10.0, 7.5, 0.1)
    research = st.radio('Research Experience', [0, 1])
    if st.button('Predict'):
        admission_chance = predict_admission(gre, toefl, universityrating, sop, lor, cgpa, research)
        admission_chance = map_to_percentage(admission_chance)
        st.success(f'Your chance of admission is {admission_chance:.2f}%.')
        
if __name__ == '__main__':
    main()
