import streamlit as st
from multiapp import MultiApp
from apps import home, pred, data, model

st.set_page_config(page_title='Stroke Prediction using ML - Mini-Project for 19CS601', page_icon = 'favicon.png', initial_sidebar_state = 'auto')

# Hide Streamlit brandings
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

app = MultiApp()

app.add_app("Home", home.app)
app.add_app("Prediction Service", pred.app)
app.add_app("Dataset Overview", data.app)
app.add_app("Model Overview", model.app)

with st.sidebar:    
    sess = app.run()

app.view(sess)