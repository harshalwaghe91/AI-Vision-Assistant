
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Settings
st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load Data
df = pd.read_csv("object_count.csv")

# Sidebar
st.sidebar.title("🤖 AI Vision Assistant")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Detection Report", "About"]
)

# ==========================
# Dashboard Page
# ==========================

if page == "Dashboard":

    st.markdown("""
    # 🤖 AI Vision Assistant

    ### Real-Time Object Detection & Analytics Dashboard
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Objects",
        int(df["Count"].sum())
    )

    col2.metric(
        "Unique Objects",
        len(df)
    )

    col3.metric(
        "Top Object",
        df.loc[df["Count"].idxmax(), "Object"]
    )

    st.subheader("📊 Object Count Analysis")

    st.bar_chart(
        df.set_index("Object")["Count"]
    )

    st.subheader("🥧 Object Distribution")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.pie(
        df["Count"],
        labels=df["Object"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# ==========================
# Detection Report Page
# ==========================

elif page == "Detection Report":

    st.title("📋 Detection Report")

    st.dataframe(df)

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Report",
        csv,
        "object_report.csv",
        "text/csv"
    )

# ==========================
# About Page
# ==========================

elif page == "About":

    st.title("ℹ️ About Project")

    st.markdown("""
    ## AI Vision Assistant

    This project uses:

    - YOLOv11
    - OpenCV
    - Streamlit
    - Python

    ### Features

    ✅ Object Detection

    ✅ Object Counting

    ✅ Analytics Dashboard

    ✅ Report Generation

    ### Developer

    Harshal Waghe
    """)



    
