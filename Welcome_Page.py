import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
img = Image.open('nit.png')
st.write("# Welcome My Network Intrusion Detection Demo! 👋")

st.sidebar.success("Select a testing method above")

st.markdown(
    """
    This is a network intrusion detection demo.
    It identifies network intrusion by using machine learning.
    You have the option of  selecting a testing method from the sidebar.
    ### Thanks for using this demo!
"""
)
st.image(img, width=700)