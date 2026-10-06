import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=":newspaper:"
)

# Custom CSS for aesthetics
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        font-family: 'Arial', sans-serif;
        animation: fadeIn 1s;
    }
    .reportview-container {
        padding: 1rem 2rem;
    }
    .stTextInput {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
        margin-bottom: 20px;
        animation: slideInDown 1s;
    }
    .btn-group button {
        background-color: #008CBA; /* Blue */
        border: 1px solid #008CBA;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .btn-group button:hover {
        background-color: #005f73;
        transform: scale(1.05);
    }
    .btn-group {
        animation: fadeInUp 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInDown {
        from {
            transform: translateY(-50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    @keyframes fadeInUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with Buttons
st.markdown("<div class='title'>✨ Fake News Detection System</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="btn-group" style="text-align: center;">
        <button>Home</button>
        <button>About</button>
        <button>Contact</button>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 2])

# Left Column: Text Input
with col1:
    st.subheader("Enter the News Article:")
    news_article = st.text_area(
        "", "Enter your news article here...", height=300
    )
    if st.button("Analyze Article"):
        st.success("Article submitted for analysis!")

# Right Column: Results
with col2:
    st.subheader("Results")
    
    # Top Section: Decision
    st.markdown("### Decision")
    st.info("**Prediction:** Fake News")  # Placeholder decision

    # Middle Section: Technical Explanation
    st.markdown("### Explanation (Technical)")
    sns.set(style="whitegrid")

    # Graphs in a Grid Layout
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    plt.subplots_adjust(hspace=0.4, wspace=0.7)

    # Graph 1: Feature Importance
    data = pd.DataFrame({"Feature": ["Sensationalism", "Misleading Words", "Factual Content"],
                         "Importance": [0.8, 0.7, 0.4]})
    sns.barplot(x="Importance", y="Feature", data=data, palette="coolwarm", ax=axes[0, 0])
    axes[0, 0].set_title("Feature Importance", fontsize=10)
    axes[0, 0].set_xlabel("Impact on Prediction", fontsize=8)
    axes[0, 0].set_ylabel("Feature", fontsize=8)

    # Graph 2: Confidence Distribution
    confidence_data = pd.DataFrame({"Confidence": [0.6, 0.7, 0.9, 0.8, 0.65], "Category": ["Fake", "Real", "Fake", "Real", "Fake"]})
    sns.boxplot(x="Category", y="Confidence", data=confidence_data, palette="viridis", ax=axes[0, 1])
    axes[0, 1].set_title("Confidence Distribution", fontsize=10)
    axes[0, 1].set_xlabel("Category", fontsize=8)
    axes[0, 1].set_ylabel("Confidence", fontsize=8)

    # Graph 3: Word Impact Heatmap
    heatmap_data = pd.DataFrame({"Word": ["Fraud", "Breaking", "Shocking", "Verify", "Credible"],
                                 "Impact": [0.9, 0.8, 0.7, 0.2, 0.1]})
    sns.heatmap(heatmap_data.set_index("Word"), annot=True, cmap="coolwarm", cbar=False, fmt=".1f", ax=axes[0, 2])
    axes[0, 2].set_title("Word Impact Heatmap", fontsize=10)

    # Graph 4: Keyword Frequency
    keyword_data = pd.DataFrame({"Keyword": ["Election", "Fraud", "Breaking", "Vaccine", "Scam"],
                                 "Frequency": [15, 25, 10, 30, 20]})
    sns.barplot(x="Frequency", y="Keyword", data=keyword_data, palette="mako", ax=axes[1, 0])
    axes[1, 0].set_title("Keyword Frequency", fontsize=10)
    axes[1, 0].set_xlabel("Frequency", fontsize=8)
    axes[1, 0].set_ylabel("Keyword", fontsize=8)

    # Graph 5: Sentiment Analysis Distribution
    sentiment_data = pd.DataFrame({"Sentiment": ["Positive", "Negative", "Neutral"], "Count": [12, 35, 18]})
    sns.barplot(x="Sentiment", y="Count", data=sentiment_data, palette="Set2", ax=axes[1, 1])
    axes[1, 1].set_title("Sentiment Analysis", fontsize=10)
    axes[1, 1].set_xlabel("Sentiment", fontsize=8)
    axes[1, 1].set_ylabel("Count", fontsize=8)

    # Graph 6: Correlation Heatmap
    corr_data = pd.DataFrame({"Feature1": [1, 0.8, 0.6], "Feature2": [0.8, 1, 0.4], "Feature3": [0.6, 0.4, 1]})
    sns.heatmap(corr_data, annot=True, cmap="Blues", ax=axes[1, 2])
    axes[1, 2].set_title("Feature Correlation", fontsize=10)

    st.pyplot(fig)

    st.markdown(
        "- **Sensationalism:** High impact on classification as fake news\n"
        "- **Misleading Words:** Phrases contribute to misinformation\n"
        "- **Factual Content:** Lower contribution due to minimal evidence"
    )

    # Bottom Section: Layman Explanation
    st.markdown("### Explanation (For Laymen)")
    st.write(
        "The system found that the article contains a lot of sensational words and misleading phrases, "
        "which strongly indicate that it might be fake. It also lacks sufficient factual information to "
        "support its claims."
    )


    st.markdown(
        """
        #### What does this mean?
        - Sensational words like "shocking," "hidden," and "exclusive" are often found in fake news.
        - Misleading phrases are attempts to make claims without evidence.
        - Articles without verified facts are likely to be classified as fake.
        """
    )

# Footer
st.markdown(
    "<hr style='border: 1px solid #ccc;'>",
    unsafe_allow_html=True
)
st.markdown(
    "<center>© 2024 Fake News Detection System. All rights reserved.</center>",
    unsafe_allow_html=True
)
