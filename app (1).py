
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Vision Assistant Dashboard")

df = pd.read_csv("object_count.csv")

st.subheader("Object Detection Report")
st.dataframe(df)

st.subheader("Object Counts")
st.bar_chart(df.set_index("Object"))
