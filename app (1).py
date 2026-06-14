
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os # Import os module to check for file existence

# Page Settings
st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="🤖",
    layout="wide"
)

# Load Data
# Check if object_count.csv exists before loading
if os.path.exists("object_count.csv"):
    df = pd.read_csv("object_count.csv")
else:
    df = pd.DataFrame({"Object": [], "Count": []}) # Create an empty DataFrame if file not found

# Sidebar
st.sidebar.title("🤖 AI Vision Assistant")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Detection Report", "Real-Time Detection", "About"]
)

# ==========================
# Dashboard Page
# ==========================

if page == "Dashboard":

    st.markdown("""
    # 🤖 AI Vision Assistant

    ### Real-Time Object Detection & Analytics Dashboard
    """)

    if not df.empty:
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
    else:
        st.info("No object detection data available yet. Please process a video on the 'Real-Time Detection' page.")

# ==========================
# Detection Report Page
# ==========================

elif page == "Detection Report":

    st.title("📋 Detection Report")

    if not df.empty:
        st.dataframe(df)

        csv = df.to_csv(index=False)

        st.download_button(
            "📥 Download Report",
            csv,
            "object_report.csv",
            "text/csv"
        )
    else:
        st.info("No object detection data available yet.")

# ==========================
# Real-Time Detection Page
# ==========================

elif page == "Real-Time Detection":

    st.title("🎥 Real-Time Object Detection")

    st.markdown("""
    This page demonstrates the real-time object detection capabilities using YOLOv11.
    Due to environment limitations for live camera feeds, we process uploaded videos.
    """)

    st.subheader("Upload Video for Processing")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.success("Video uploaded successfully!")
        st.info("To process this video, please run the dedicated YOLO processing cell in the notebook. Once processed, the output will appear here.")
        st.video(uploaded_file)

    st.subheader("Previously Processed Video")
    processed_video_path = "ai_output.mp4"
    if os.path.exists(processed_video_path):
        with open(processed_video_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.success("Displaying processed video from previous runs.")
    else:
        st.info("No processed video (`ai_output.mp4`) found. Please upload a video and run the processing code in the notebook.")

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


import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

st.title("🤖 Real-Time Object Detection")

# Load YOLO model
model = YOLO("yolo11n.pt")

class YOLOProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        results = model(img)

        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(
            annotated_frame,
            format="bgr24"
        )

webrtc_streamer(
    key="yolo",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
