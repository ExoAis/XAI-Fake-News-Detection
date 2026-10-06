import streamlit as st
import time

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon=":newspaper:"
)

# Corrected Typewriter effect function
def typewriter(text: str, speed: int, container):
    tokens = text.split()
    container = st.empty()  # Empty container to hold the dynamic text
    for index in range(len(tokens) + 1):
        curr_full_text = " ".join(tokens[:index])
        container.markdown(curr_full_text, unsafe_allow_html=True)
        time.sleep(1 / speed)

# Custom CSS for the design of the landing page
st.markdown(
    """
    <style>
    .main {
        background: #000000;
        font-family: 'Arial', sans-serif;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        flex-direction: column;
        text-align: center;
        color: white;
    }

    .hero-container {
        padding: 40px;
        max-width: 1000px;
        animation: fadeIn 2s ease-out;
    }

    .hero-container h1 {
        font-size: 50px;
        font-weight: bold;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #fff;
        animation: slideInDown 1s ease-out;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 15px;
    }

    .hero-container p {
        font-size: 20px;
        margin-bottom: 30px;
        line-height: 1.6;
        color: #fff;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .cta-button {
        background-color: #ffffff;
        padding: 15px 30px;
        border-radius: 10px;
        color: black;
        font-size: 20px;
        text-decoration: none;
        display: inline-block;
        cursor: pointer;
        transition: background-color 0.3s, transform 0.3s;
        animation: fadeInUp 1s ease-out;
    }

    .cta-button:hover {
        background-color: #ffffff;
        transform: scale(1.1);
    }

    .feature-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 30px;
        padding: 20px;
        animation: fadeIn 2s ease-out;
        margin-top: 50px;
    }

    .feature-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(0, 0, 0, 0.3));
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .feature-box h3 {
        font-size: 24px;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }

    .feature-box p {
        font-size: 18px;
        color: #fff;
        line-height: 1.6;
    }

    .feature-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
    }

    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: white;
        font-weight: bold;
        background-color: rgba(0, 0, 0, 0.5);
        padding: 10px 0;
    }

    .footer a {
        color: #ff6347;
        text-decoration: none;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes slideInDown {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section with typewriter effect
st.markdown('<div class="hero-container">', unsafe_allow_html=True)

# Title & Description with typewriter effect
st.markdown("<h1>Fake News Detection System</h1>", unsafe_allow_html=True)
text_container = st.empty()  # Create an empty container
typewriter("Detect and eliminate misinformation in real-time with cutting-edge technology that analyzes and identifies fake news articles.", 30, text_container)

# Call-to-action Button
st.markdown('<a href="#features" class="cta-button">Discover More</a>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Features Section
st.markdown('<div id="features" class="feature-container">', unsafe_allow_html=True)

# Feature 1
cont = st.markdown('<div class="feature-box"><h3>Real-time Detection</h3>', unsafe_allow_html=True)
typewriter("<div class=feature-box><h3>Instantly analyze news articles for credibility, identifying fake news and misleading content. </h3>", 30, cont)
st.markdown('</div>', unsafe_allow_html=True)

# Feature 2
cont = st.markdown('<div class="feature-box"><h3>Advanced Algorithm</h3>', unsafe_allow_html=True)
typewriter("<div class='feature-box'><h3>Leverage deep learning and NLP models for accurate, data-driven predictions of fake news.</h3>", 30, cont)
st.markdown('</div>', unsafe_allow_html=True)

# Feature 3
cont = st.markdown('<div class="feature-box"><h3>User-friendly Interface</h3>', unsafe_allow_html=True)
typewriter("<div class='feature-box'><h3>Easy to use for anyone, just input a news article, and get an immediate verdict on its authenticity.</h3>", 30, cont)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Footer with copyright (fixed to the bottom)
st.markdown(
    """
    <div class="footer">
        <p>© 2024 Fake News Detection System. All rights reserved. <a href="#">Privacy Policy</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
