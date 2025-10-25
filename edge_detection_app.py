import streamlit as st
import cv2
import numpy as np
from PIL import Image

# streamlit page configuration
st.set_page_config(
    page_title="Edge Detection Studio",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# css - taken help from internet :
st.markdown(
    """
    <style>

    body {
        background-color: #0B0C10;
        color: #EAEAEA;
        font-family: 'Poppins', sans-serif;
    }


    header, footer {
        visibility: visible !important;
    }


    [data-testid="collapsedControl"] {
        color: white !important;
        background-color: rgba(102, 252, 241, 0.2) !important;
        border: 1px solid rgba(102,252,241,0.4);
        border-radius: 6px;
        top: 12px;
        left: 12px;
        position: fixed;
        z-index: 9999;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(102, 252, 241, 0.4) !important;
        transform: scale(1.05);
    }


    .top-header {
        background: linear-gradient(90deg, #0F2027, #203A43, #2C5364);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 6px 30px rgba(2,8,23,0.6);
    }
    .top-header h1 { color: #66FCF1; margin:0; font-weight:700; letter-spacing:0.6px; }
    .top-header p { color: #C5C6C7; margin:4px 0 0 0; font-size:0.95rem; }


    .upload-box {
        border: 2px dashed rgba(102, 252, 241, 0.25);
        border-radius: 12px;
        padding: 36px;
        text-align: center;
        background: linear-gradient(180deg, rgba(255,255,255,0.85), rgba(240,240,240,0.75));
        transition: all 0.25s ease;
        color: #000000;
    }
    .upload-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(32,50,60,0.35);
    }


    [data-testid="stSidebar"] {
        background-color: #0E1519;
        border-right: 1px solid rgba(102,252,241,0.04);
        color: white !important;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

  
    div[data-baseweb="select"] * {
        color: black !important;
    }


    .stButton>button {
        background: linear-gradient(90deg, #45A29E, #66FCF1);
        color: #061014;
        font-weight: 700;
        border-radius: 10px;
        padding: 8px 16px;
        border: none;
    }
    .stButton>button:hover { transform: scale(1.03); }

    .footer {
        text-align:center;
        color:#8c98a0;
        margin-top: 22px;
        font-size:0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="top-header">
        <h1> Edge Detection App</h1>
        <p>Experiment interactively with Canny, Sobel, and Laplacian detectors. Control parameter from the sidebar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# sidebar
st.sidebar.markdown("##  Controls")
st.sidebar.markdown("Choose detector and tune parameters")

method = st.sidebar.selectbox("Detection method", ("Canny", "Sobel", "Laplacian"))

auto_update = st.sidebar.checkbox("Auto update (real-time)", value=True)
apply_btn = st.sidebar.button("Apply")


st.sidebar.markdown("---")
st.sidebar.markdown(" Detector Parameters")

if method == "Canny":
    canny_low = st.sidebar.slider("Lower threshold", 0, 255, 100)
    canny_high = st.sidebar.slider("Upper threshold", 0, 255, 200)
    gauss_ksize = st.sidebar.slider("Gaussian kernel size (odd)", 1, 31, 5, step=2)
    gauss_sigma = st.sidebar.slider("Gaussian sigma", 0.0, 10.0, 1.0, step=0.1)

elif method == "Sobel":
    sobel_ksize = st.sidebar.slider("Kernel size (odd)", 1, 7, 3, step=2)
    sobel_dx = st.sidebar.selectbox("Gradient X ?", (0, 1), index=1)
    sobel_dy = st.sidebar.selectbox("Gradient Y ?", (0, 1), index=0)
    sobel_comb = st.sidebar.checkbox("Combine X & Y magnitude", value=True)

elif method == "Laplacian":
    lap_ksize = st.sidebar.slider("Kernel size (odd)", 1, 7, 3, step=2)
    lap_scale = st.sidebar.slider("Scale", 1.0, 5.0, 1.0, step=0.1)


# inverting image theme
st.sidebar.markdown("---")
st.sidebar.markdown("Output")
invert = st.sidebar.checkbox("Invert output (white edges on black)", value=False)

# main body
uploaded_file = st.file_uploader(" Upload an image (JPG/PNG/BMP)", type=["jpg", "jpeg", "png", "bmp"])

def process_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    result = np.zeros_like(gray)

# if user choose canny:
    if method == "Canny":
        blurred = cv2.GaussianBlur(gray, (gauss_ksize, gauss_ksize), sigmaX=gauss_sigma)
        result = cv2.Canny(blurred, canny_low, canny_high)
    elif method == "Sobel":
        gx = gy = None
        if sobel_dx:
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
            gx = cv2.convertScaleAbs(gx)
        if sobel_dy:
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
            gy = cv2.convertScaleAbs(gy)
        if gx is not None and gy is not None and sobel_comb:
            mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
            result = cv2.convertScaleAbs(mag)
        else:
            result = gx if gx is not None else gy
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=lap_ksize, scale=lap_scale)
        result = cv2.convertScaleAbs(lap)

    if invert:
        result = 255 - result
    return result

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_img)
    edges = process_image(img_rgb)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(" Original")
        st.image(pil_img, use_column_width=True)
    with col2:
        st.markdown(f" {method} Result")
        st.image(edges, use_column_width=True, clamp=True)

# adding download button-take a little help from internet (youtube)
    st.download_button(
        label=" Download Processed Image",
        data=cv2.imencode(".png", edges)[1].tobytes(),
        file_name=f"{method}_edges.png",
        mime="image/png"
    )
else:
    # take help for uploading image :
    st.markdown(
        """
        <div class="upload-box">
            <h3> Upload an image to begin</h3>
            <p><b>Supported formats:</b> JPG, PNG, BMP.<br>Use the sidebar to select detector and tune parameters.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

