import streamlit as st
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom imports 
from multipage import MultiPage
from src import model, eda, inference # import your pages here

# Create an instance of the app 
app = MultiPage()

# Title of the main page
st.title("Crime Analytics")

# Add all your applications (pages) here
app.add_page("Data Visualization", eda.app)
app.add_page("Model Creation", model.app)
app.add_page("Crime Prediction", inference.app)

# The main app
app.run()