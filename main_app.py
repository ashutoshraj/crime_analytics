import streamlit as st
st.set_page_config(
    page_title="Crime Analytics Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    # initial_sidebar_state="expanded"
)

# Custom imports 
from multipage import MultiPage
from PIL import  Image
import numpy as np
from src import model, eda, inference # import your pages here

# Create an instance of the app 
app = MultiPage()

display = Image.open('Logo.jpg')
display = np.array(display)

# Title of the main page
col1, col2 = st.columns(2)
col1.title("Crime Analytics")
col1.write(
    """Crime in society has been on the rise globally, particularly
     pronounced in urban areas. Common causes include increased urbanization,
      nuclear families, population densities, education, income disparities and unemployment.
       Our project collects crime data and data on predictors from several sources to provide
        analytics and a model to predict crime in the US."""
)  # description and instructions

col2.image(display, width = 300)

# Add all your applications (pages) here
app.add_page("Data Visualization", eda.app)
app.add_page("Model Creation", model.app)
app.add_page("Crime Prediction", inference.app)

# The main app
app.run()