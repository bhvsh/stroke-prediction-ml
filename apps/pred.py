import streamlit as st
import lightgbm
import pickle
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
model = pickle.load(open("apps\models\gbm\gbm-model-pickle.sav", 'rb'))
scaler = pickle.load(open("apps\models\gbm\gbm-scaler.sav", 'rb'))

def app():
    with st.sidebar:
        st.title('Stroke Prediction using Machine Learning')

        st.write('This model which predicts whether a patient is likely to get a stroke based on the parameters like gender, age various diseases and smoking status.')
        st.markdown('_For Machine Learning - 19CS601_')

        st.write('It may take a few moments to complete this survey.')

    with st.container():
        st.subheader('Stage 1: Personal Questions')

        ch_gender = st.selectbox(
            'Gender: ',
            ('Male', 'Female', 'Others'))
            
        ch_age = st.number_input('Age: ',min_value=0, max_value=150, value=18,step=1)

        ch_restype = st.radio(
            'Residence Type: ',
            ('Urban', 'Rural'))

        ch_marital = st.radio(
            'Did you ever get married? ',
            ('Yes', 'No'))

        ch_worktype = st.selectbox(
        'Work type: ',
        ('I\'m a child.', 'I\'m self employed', 'Working for the Private.','Working for the Government.','Never worked for anyone.'))

        st.subheader('Stage 2: Health Questions')

        ch_height = st.number_input('Height (in m): ',min_value=0.0, max_value=500.0, value=175.0,step=0.1)

        ch_weight = st.number_input('Weight (in kg): ',min_value=0.0, max_value=5000.0, value=75.0,step=0.01)
        
        calc_bmi  = ch_weight / (ch_height/100)**2

        ch_bmi = st.number_input('BMI: (Optional)',min_value=0.0, max_value=60.0, value=calc_bmi,step=0.01)

        ch_agl = st.number_input('Average Glucose Level (in mg/dL): ',min_value=50.0, max_value=300.0, value=50.0,step=0.01)

        ch_smokingstat = st.selectbox(
            'Smoking status: ',
            ('Never smoked', 'Formerly smoked', 'I\'m an active smoker','I prefer not to speak'))

        st.write('Are you currently suffering from these diseases?')

        ch_hypertn = st.checkbox('Hypertension')

        ch_hearttn = st.checkbox('Heart Disease')

        submit = st.button('Submit')

        if submit:
            
            ch_gender = 0 if ch_gender=="Female" else 1 if ch_gender=="Male" else 2
            ch_marital =  1 if ch_marital=="Yes" else 0
            ch_worktype = 1 if ch_worktype=="Never worked for anyone." else 4 if ch_worktype=="I\'m a child." else 3 if ch_worktype=="I\'m self employed" else 2 if ch_worktype=="Working for the Private." else 0
            ch_restype = 1 if ch_restype=="Urban" else 1
            ch_smokingstat = 3 if ch_smokingstat=="I\'m an active smoker" else 1 if ch_smokingstat=="Formerly smoked" else 2 if ch_smokingstat=="Never smoked" else 0
            ch_hypertn =  0 if ch_hypertn==False else 1 if ch_hypertn==True else 999
            ch_hearttn =  0 if ch_hearttn==False else 1 if ch_hearttn==True else 999

            input = scaler.transform([[ch_gender,ch_age,ch_hypertn,ch_hearttn,ch_marital,ch_worktype,ch_restype,ch_agl,ch_bmi,ch_smokingstat]])

            prediction = model.predict(input)
            predictval = model.predict_proba(input)
            
            with st.expander("Results"):
                if prediction==0:
                    str_result = 'The model predicts that with the probability of %.2f%%, you won\'t be suffering from stroke in the future.'%(predictval[0][0]*100)
                    st.success(str_result)
                    st.write("""
                        The best way to help prevent a stroke is to eat a healthy diet, exercise regularly, and avoid smoking and drinking too much alcohol.
                        These lifestyle changes can reduce your risk of problems like:
                        - arteries becoming clogged with fatty substances (atherosclerosis)
                        - high blood pressure
                        - high cholesterol levels
                        If you have already had a stroke, making these changes can help reduce your risk of having another stroke in the future.

                    """)
                    st.write("Source: [National Health Service (NHS) - United Kingdom](https://www.nhs.uk/conditions/stroke/prevention/)")

                elif prediction==1:
                    str_result = 'The model predicts that with the probability of %.2f%%, you will be suffering from stroke in the future.'%(predictval[0][1]*100)
                    st.error(str_result)
                    if predictval[0][1] >= 0.85:
                        st.subheader("Please seek medical attention as early as possible to mitigate the stroke disease.")
                    st.write("""
                        The best way to help prevent a stroke is to eat a healthy diet, exercise regularly, and avoid smoking and drinking too much alcohol.
                        These lifestyle changes can reduce your risk of problems like:
                        - arteries becoming clogged with fatty substances (atherosclerosis)
                        - high blood pressure
                        - high cholesterol levels
                        If you have already had a stroke, making these changes can help reduce your risk of having another stroke in the future.

                    """)
                    st.write("Source: [National Health Service (NHS) - United Kingdom](https://www.nhs.uk/conditions/stroke/prevention/)")

                else:
                    st.error('NaN: Unexpected error')
                    st.markdown("Debug: Selected input:")
                    st.code(input)