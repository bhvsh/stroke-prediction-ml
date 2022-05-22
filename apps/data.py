import streamlit as st
import pandas as pd

def app():
    with st.sidebar:
        st.title('Stroke Prediction using Machine Learning')

        st.write('This model which predicts whether a patient is likely to get a stroke based on the parameters like gender, age various diseases and smoking status.')
        st.markdown('_For Machine Learning - 19CS601_')
    
    st.title('Dataset Overview')

    st.write("The following is the DataFrame of the healthcare dataset for stroke prediction.")
    st.write('This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.')

    st.markdown('Dataset by Federico Soriano Palacios ([__fedesoriano__](https://www.kaggle.com/fedesoriano) on Kaggle)')
    st.markdown('Source: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset')

    df = pd.read_csv("dataset\healthcare-dataset-stroke-data.csv")
    df['hypertension'] = df['hypertension'].map({0:"No", 1:"Yes"})
    df['heart_disease'] = df['heart_disease'].map({0:"No", 1:"Yes"})
    df['stroke'] = df['stroke'].map({0:"No", 1:"Yes"})
    st.write(df)
