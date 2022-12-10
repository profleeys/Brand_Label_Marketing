import streamlit as st

def long_computation():
    import time
    time.sleep(0.5)

st.header("TEST DOWNLOAD BUTTON")

with st.spinner("Running a long computation.."):
    long_computation()

st.download_button("Download", "a")