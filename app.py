import streamlit as st
import pandas as pd
import os
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

url = "https://cdn2.iconfinder.com/data/icons/databases-and-cloud-technology/48/24-512.png"

with st.sidebar:
    st.image(url)
    st.title("AutoEDA")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("This app helps you explore your data using Streamlit, Pandas Profiling and PyCaret.")

if os.path.exists("source_data.csv"):
    df = pd.read_csv("source_data.csv", index_col=None)

if choice == "Upload":
    st.title("Upload dataset for modelling")
    file = st.file_uploader("Upload your dataset here", type=["csv", "xlsx"])
    if file is not None:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("source_data.csv", index=None)
        st.dataframe(df)

elif choice == "Profiling":
    st.title("Automated exploratory data analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)

elif choice == "Modelling":
    st.title("Automated modelling")
    target = st.selectbox("Select target column", df.columns)
    if st.button("Run modelling"):
        setup(df, target=target, silent=True)
        setup_df = pull()
        st.info("These are the modelling settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("These are the ML models and their scores")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")

elif choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the model", f, "trained_model.pkl")
