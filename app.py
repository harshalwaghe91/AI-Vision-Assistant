import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av

st.title("Real-Time Object Detection")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            results = model.predict(img, verbose=False)
            annotated = results[0].plot()

            return av.VideoFrame.from_ndarray(
                annotated,
                format="bgr24"
            )
        except Exception as e:
            print(e)
            return frame

webrtc_streamer(
    key="camera",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
