import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Agri AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main page */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #eafaf1 0%, #f4f9f6 35%, #eef3f0 100%);
    }

    #MainMenu, footer {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b3d24 0%, #14532d 100%);
    }

    section[data-testid="stSidebar"] * {
        color: #eafaf1 !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Hero */
    .hero {
        position: relative;
        padding: 3rem 2rem;
        border-radius: 28px;
        background: linear-gradient(120deg, #0f5132 0%, #198754 55%, #2fbf71 100%);
        color: white;
        text-align: center;
        margin-bottom: 2.2rem;
        box-shadow: 0 20px 45px -15px rgba(15, 81, 50, 0.55);
        overflow: hidden;
    }

    .hero::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: rgba(255,255,255,0.08);
        border-radius: 50%;
    }

    .hero::after {
        content: "";
        position: absolute;
        bottom: -80px;
        left: -40px;
        width: 260px;
        height: 260px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }

    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }

    .hero p {
        font-size: 1.15rem;
        opacity: 0.92;
        font-weight: 400;
        max-width: 620px;
        margin: 0 auto;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        background: rgba(255,255,255,0.16);
        backdrop-filter: blur(6px);
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* Section headers */
    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.35rem;
        color: #14532d;
        margin-bottom: 0.6rem;
    }

    /* Upload / analysis panels */
    .panel {
        background: white;
        border-radius: 22px;
        padding: 1.6rem;
        border: 1px solid #dfeee3;
        box-shadow: 0 10px 30px -12px rgba(20, 83, 45, 0.12);
        height: 100%;
    }

    /* Result card */
    .result-card {
        padding: 1.8rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #ffffff, #f2fbf5);
        border: 1px solid #cdeedc;
        box-shadow: 0 14px 35px -14px rgba(20, 108, 67, 0.28);
        margin-top: 1.2rem;
        animation: fadeIn 0.5s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .disease-name {
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #146c43;
    }

    .confidence {
        font-size: 1.15rem;
        font-weight: 600;
        color: #3a3a3a;
        margin-top: 0.3rem;
    }

    .confidence-value {
        color: #198754;
        font-weight: 800;
    }

    /* Info cards */
    .info-card {
        padding: 1.4rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e1e8e3;
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 8px 20px -12px rgba(0,0,0,0.08);
    }

    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 30px -14px rgba(20, 108, 67, 0.25);
    }

    .info-card h3 {
        font-family: 'Poppins', sans-serif;
        color: #146c43;
        margin-bottom: 0.4rem;
        font-size: 1.1rem;
    }

    .info-card p {
        color: #555;
        font-size: 0.93rem;
        margin: 0;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1rem !important;
        background: linear-gradient(135deg, #198754, #146c43) !important;
        border: none !important;
        box-shadow: 0 10px 22px -8px rgba(20, 108, 67, 0.45);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px -8px rgba(20, 108, 67, 0.55);
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #198754, #2fbf71);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px !important;
        border: 1.5px dashed #9cd3af !important;
        background: #f7fdf9 !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 3rem;
        padding: 1.2rem;
        border-top: 1px solid #e1e8e3;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Hero section
# --------------------------------------------------

st.markdown(
"""
<div class="hero">
<h1>🌱 Smart Agri AI</h1>
<p>AI-powered crop disease detection using Computer Vision and Deep Learning</p>
<div class="hero-badge">🔬 ResNet18 · 38 Classes</div>
</div>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ System")

    st.write("**Model:** ResNet18")

    st.write("**Input:** Plant image")

    st.write("**Image size:** 224 × 224")

    st.write("**Classes:** 38")

    st.divider()

    st.subheader("🔌 API Status")

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:

            st.success("API Online")

        else:

            st.error("API Error")

    except requests.exceptions.RequestException:

        st.error("API Offline")

        st.caption(
            "Start FastAPI before using predictions."
        )


# --------------------------------------------------
# Main content
# --------------------------------------------------

left_column, right_column = st.columns(
    [1, 1],
    gap="large",
)


# --------------------------------------------------
# Upload section
# --------------------------------------------------

with left_column:

    st.markdown('<div class="section-title">📤 Upload Plant Image</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a plant leaf.",
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded image",
            use_container_width=True,
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )

        st.caption(
            f"Size: {uploaded_file.size / 1024:.1f} KB"
        )

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Prediction section
# --------------------------------------------------

with right_column:

    st.markdown('<div class="section-title">🔬 Disease Analysis</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if uploaded_file is None:

        st.info(
            "Upload a plant image to begin analysis."
        )

    else:

        st.write(
            "Ready to analyze the uploaded image."
        )

        analyze_button = st.button(
            "🔍 Analyze Plant",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:

            with st.spinner(
                "AI model is analyzing the image..."
            ):

                try:

                    uploaded_file.seek(0)

                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }

                    response = requests.post(
                        f"{API_URL}/predict",
                        files=files,
                        timeout=60,
                    )

                    if response.status_code == 200:

                        result = response.json()

                        prediction = result[
                            "prediction"
                        ]

                        disease = prediction[
                            "disease"
                        ]

                        confidence = prediction[
                            "confidence"
                        ]

                        confidence_percent = (
                            confidence * 100
                        )

                        st.success(
                            "Analysis completed successfully!"
                        )

                        st.markdown(
f"""
<div class="result-card">
<div class="disease-name">🌿 {disease}</div>
<br>
<div class="confidence">Confidence: <span class="confidence-value">{confidence_percent:.2f}%</span></div>
</div>
""",
                            unsafe_allow_html=True,
                        )

                        st.progress(
                            confidence
                        )

                    else:

                        st.error(
                            f"API returned error "
                            f"{response.status_code}"
                        )

                        try:

                            error_detail = response.json()

                            st.json(error_detail)

                        except Exception:

                            st.write(
                                response.text
                            )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Could not connect to the FastAPI server."
                    )

                    st.info(
                        "Make sure FastAPI is running on "
                        "http://127.0.0.1:8000"
                    )

                except requests.exceptions.Timeout:

                    st.error(
                        "The prediction request timed out."
                    )

                except Exception as error:

                    st.error(
                        "Something went wrong."
                    )

                    st.exception(error)

    st.markdown('</div>', unsafe_allow_html=True)


# --------------------------------------------------
# Information section
# --------------------------------------------------

st.divider()

st.markdown('<div class="section-title">🚀 About Smart Agri AI</div>', unsafe_allow_html=True)

info1, info2, info3 = st.columns(3, gap="medium")


with info1:

    st.markdown(
"""
<div class="info-card">
<h3>🧠 Deep Learning</h3>
<p>Uses a ResNet18 computer vision model trained to classify plant diseases.</p>
</div>
""",
        unsafe_allow_html=True,
    )


with info2:

    st.markdown(
"""
<div class="info-card">
<h3>🌿 Plant Health</h3>
<p>The system analyzes an uploaded plant image and predicts the most likely class.</p>
</div>
""",
        unsafe_allow_html=True,
    )


with info3:

    st.markdown(
"""
<div class="info-card">
<h3>⚡ Fast API</h3>
<p>Streamlit communicates with a separate FastAPI backend for model inference.</p>
</div>
""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.markdown(
"""
<div class="footer">
⚠️ <b>Disclaimer:</b> This system provides an AI-based prediction for educational and research purposes. It should not replace professional agricultural advice.
<br><br>
🌱 Smart Agri AI &nbsp;•&nbsp; Computer Vision &nbsp;•&nbsp; PyTorch &nbsp;•&nbsp; FastAPI
</div>
""",
    unsafe_allow_html=True,
)