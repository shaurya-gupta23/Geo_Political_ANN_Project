import streamlit as st
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="GeoMarket AI",
    page_icon="🌍",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background: linear-gradient(
        135deg,
        #0F172A,
        #111827,
        #1E293B
    );
}

.main-title{
    font-size:55px;
    font-weight:bold;
    color:#38BDF8;
}

.sub-title{
    font-size:22px;
    color:white;
}

[data-testid="metric-container"]{
    background:#1E293B;
    border-radius:15px;
    padding:15px;
    border:1px solid #334155;
}

.result-up{
    background:#14532D;
    padding:20px;
    border-radius:15px;
    color:white;
}

.result-down{
    background:#7F1D1D;
    padding:20px;
    border-radius:15px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load("geopolitical_model.pkl")
tfidf = joblib.load("tfidf.pkl")

analyzer = SentimentIntensityAnalyzer()

# ==================================================
# HELPER FUNCTION
# ==================================================

def count_keywords(text, keywords):
    text = text.lower()
    return sum(text.count(word) for word in keywords)

war_words = [
    "war",
    "attack",
    "missile",
    "conflict",
    "invasion",
    "military",
    "troops"
]

oil_words = [
    "oil",
    "crude",
    "opec",
    "petroleum"
]

trade_words = [
    "tariff",
    "sanction",
    "trade war",
    "export",
    "import"
]

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🌍 GeoMarket AI")

st.sidebar.markdown("""
### Geopolitical Intelligence Dashboard

#### Features

⚔ War Detection

🛢 Oil Crisis Analysis

🌐 Trade/Tariff Analysis

😊 Sentiment Analysis

📈 Stock Market Prediction

🎯 Confidence Estimation
""")

st.sidebar.markdown("---")

st.sidebar.success("Model: Random Forest")

# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<p class="main-title">🌍 GeoMarket AI Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Impact of Geopolitical Events on Stock Market Direction Using Machine Learning</p>',
    unsafe_allow_html=True
)

st.divider()

# ==================================================
# STOCK MARKET BANNER
# ==================================================

st.image(
    "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3",
    use_container_width=True
)

st.divider()

# ==================================================
# TOP METRICS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌍 Countries", "195")

with col2:
    st.metric("⚔ War Signals", "7")

with col3:
    st.metric("🛢 Oil Signals", "4")

with col4:
    st.metric("📈 ML Models", "3")

st.divider()

# ==================================================
# FEATURE CARDS
# ==================================================

st.subheader("🚀 System Capabilities")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("""
    ⚔ **War Detection**

    Detects military conflicts,
    invasions, attacks and
    geopolitical tensions.
    """)

with c2:
    st.info("""
    🛢 **Oil Market Analysis**

    Tracks crude oil,
    petroleum and
    OPEC-related signals.
    """)

with c3:
    st.info("""
    📈 **Market Forecasting**

    Predicts stock market
    direction based on
    geopolitical news.
    """)

st.divider()

# ==================================================
# NEWS INPUT
# ==================================================

st.subheader("📰 Enter Geopolitical News")

news = st.text_area(
    "",
    height=250,
    placeholder="""
Example:

Major military conflict escalates in the Middle East.
Oil prices surge after missile attacks.
Global investors become increasingly concerned.
"""
)

# ==================================================
# PREDICT
# ==================================================

if st.button("🚀 Analyze Market Impact"):

    sentiment = analyzer.polarity_scores(news)["compound"]

    war_count = count_keywords(news, war_words)
    oil_count = count_keywords(news, oil_words)
    trade_count = count_keywords(news, trade_words)

    text_features = tfidf.transform([news]).toarray()

    extra_features = np.array([
        [
            sentiment,
            war_count,
            oil_count,
            trade_count
        ]
    ])

    X = np.hstack((text_features, extra_features))

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    confidence = np.max(probabilities) * 100

    risk_score = (
        war_count * 10
        + oil_count * 5
        + trade_count * 3
    )

    st.divider()

    # ==========================================
    # RESULT
    # ==========================================

    if prediction == 1:

        st.markdown(
            f"""
            <div class="result-up">
            <h2>📈 MARKET LIKELY TO GO UP</h2>
            <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-down">
            <h2>📉 MARKET LIKELY TO GO DOWN</h2>
            <h3>Confidence: {confidence:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # ==========================================
    # ANALYTICS
    # ==========================================

    st.subheader("📊 Prediction Analytics")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "😊 Sentiment",
            round(sentiment, 3)
        )

    with m2:
        st.metric(
            "⚔ War Count",
            war_count
        )

    with m3:
        st.metric(
            "🛢 Oil Count",
            oil_count
        )

    with m4:
        st.metric(
            "🌐 Trade Count",
            trade_count
        )

    st.divider()

    # ==========================================
    # CONFIDENCE
    # ==========================================

    st.subheader("🎯 Prediction Confidence")

    st.progress(float(confidence / 100))

    st.write(
        f"Confidence Level: {confidence:.2f}%"
    )

    st.divider()

    # ==========================================
    # RISK SCORE
    # ==========================================

    st.subheader("⚠ Geopolitical Risk Score")

    st.metric(
        "Risk Score",
        risk_score
    )

    st.divider()

    # ==========================================
    # AI ANALYSIS
    # ==========================================

    st.subheader("🧠 AI News Analysis")

    if war_count > 0:
        st.warning(
            "Military conflict signals detected."
        )

    if oil_count > 0:
        st.warning(
            "Oil market disruption signals detected."
        )

    if trade_count > 0:
        st.warning(
            "Trade/Tariff related signals detected."
        )

    if (
        war_count == 0
        and oil_count == 0
        and trade_count == 0
    ):
        st.success(
            "No major geopolitical risk indicators detected."
        )

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.markdown("""
## ⚙ Technology Stack

| Component | Technology |
|------------|------------|
| NLP | TF-IDF |
| Sentiment Analysis | VADER |
| Machine Learning | Random Forest |
| Frontend | Streamlit |
| Language | Python |

---

### 📚 Project Information

**Title:** Impact of Geopolitical Events on Stock Market Direction Using Machine Learning and Sentiment Analysis

**Models Used:**
- Random Forest
- ANN
- Logistic Regression

**Developed For Academic Project Demonstration**
""")