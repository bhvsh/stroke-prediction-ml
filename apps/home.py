import streamlit as st
from PIL import Image

def app():
    with st.container():
        st.title('Stroke Prediction using Machine Learning')
        st.markdown('For _Machine Learning - 19CS601_')

        st.write('This model which predicts whether a patient is likely to get a stroke based on the parameters like gender, age various diseases and smoking status.')

        st.write('* Pick the \'Prediction Service\' to check the working of the model.')

        st.write('* Pick the \'Dataset Overview\' to know more about the dataset.')

        st.write('* Pick the \'Model Overview\' to know more about the model that we have used for predictions.')