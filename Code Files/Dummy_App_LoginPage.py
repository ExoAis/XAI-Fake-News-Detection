import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Login Page",
    layout="centered",
    initial_sidebar_state="collapsed",
    page_icon=":lock:"
)

# Custom CSS for a beautiful login page design
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
        color: white;
    }

    .h2 {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 30px;
        text-align: center;  /* Center the title */
    }
    
    .stTextInput, .stPasswordInput {
        background-color: transparent;
        border: 1px solid #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        font-size: 16px;
        color: #fff;
    }

    .stButton button {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        color: black;
        font-size: 18px;
        width: 100%;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
        margin-top: 10px;
    }

    .stButton button:hover {
        background-color: #0099FF;
    }

    .footer {
        position: fixed;
        bottom: 20px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: white;
        font-weight: bold;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# Title - Centered
st.markdown("<h2><center>Login</center></h2>", unsafe_allow_html=True)

# Username/Email input
username = st.text_input("Username / Email", placeholder="Enter your username or email")

# Password input
password = st.text_input("Password", type="password", placeholder="Enter your password")

# Login Button
if st.button("Login"):
    if username and password:
        st.success("Login successful!")
    else:
        st.error("Please enter both username and password.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer with copyright
st.markdown(
    """
    <div class="footer">
        <p>© 2024 Fake News Detection System. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
