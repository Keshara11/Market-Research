import streamlit as st
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Market Research & Competitor Analysis", layout="wide", page_icon="📊")

# -------------------------------
# Custom CSS for styling
# -------------------------------
st.markdown(
    """
    <style>
    /* Background */
    .reportview-container, .main, header, footer {background-color: #0B1D51;}
    
    /* Title and Headers */
    h1, h2, h3, h4, h5, h6 {color: white;}
    
    /* Buttons */
    div.stButton > button {background-color: #FF2E2E; color: white; border-radius: 5px;}
    
    /* Selectbox and other widgets */
    div.stSelectbox > div > div > div > input {background-color: #0B1D51; color: white;}
    
    /* Altair charts tooltip styling */
    div.vega-tooltip {background-color: black; color: white;}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("competitor_analysis_dataset.csv")
    df["industry"] = df["industry"].str.title()
    df["region"] = df["region"].str.title()
    df["sentiment_label"] = df["sentiment_label"].str.title()
    return df

df = load_data()

# -------------------------------
# Train model (cached)
# -------------------------------
@st.cache_resource
def train_model(df):
    X = df[["industry", "region"]]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ["industry", "region"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model

model = train_model(df)

# -------------------------------
# App Title
# -------------------------------
st.markdown("<h1>📊 Competitor Analysis Dashboard</h1>", unsafe_allow_html=True)

# -------------------------------
# Prediction Section
# -------------------------------
st.header("🔮 Predict Sentiment")
industry = st.selectbox("Select Industry", sorted(df["industry"].unique()))
region = st.selectbox("Select Region", sorted(df["region"].unique()))

if st.button("Predict Sentiment"):
    input_data = pd.DataFrame({"industry": [industry], "region": [region]})
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)
    st.success(f"Predicted Sentiment: {prediction}")
    st.write("### Prediction Probabilities")
    st.write(dict(zip(model.classes_, probability[0])))

# -------------------------------
# Visualizations
# -------------------------------
st.markdown("---")
st.header("📈 Competitor Insights")

# -------------------------------
# Set dark theme for Altair
# -------------------------------
alt.themes.enable("dark")

# Sentiment color map (red theme)
sentiment_colors = {
    "Positive": "#FF4C4C",
    "Neutral": "#FFAAAA",
    "Negative": "#660000"
}

# -------------------------------
# Overall Sentiment Distribution
# -------------------------------
st.subheader("Overall Sentiment Distribution")
overall = df.groupby("sentiment_label").size().reset_index(name="count")
overall_chart = alt.Chart(overall).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X("sentiment_label:N", title="Sentiment"),
    y=alt.Y("count:Q", title="Count"),
    color=alt.Color("sentiment_label:N",
                    scale=alt.Scale(domain=list(sentiment_colors.keys()),
                                    range=list(sentiment_colors.values())),
                    legend=alt.Legend(title="Sentiment")),
    tooltip=["sentiment_label", "count"]
).properties(height=400)
st.altair_chart(overall_chart, use_container_width=True)

# -------------------------------
# Sentiment by Industry
# -------------------------------
st.subheader("Sentiment Distribution by Industry")
industry_df = df.groupby(["industry", "sentiment_label"]).size().reset_index(name="count")
industry_chart = alt.Chart(industry_df).mark_bar().encode(
    x=alt.X("industry:N", sort="-y"),
    y=alt.Y("count:Q"),
    color=alt.Color("sentiment_label:N",
                    scale=alt.Scale(domain=list(sentiment_colors.keys()),
                                    range=list(sentiment_colors.values())),
                    legend=alt.Legend(title="Sentiment")),
    tooltip=["industry", "sentiment_label", "count"]
).interactive()
st.altair_chart(industry_chart, use_container_width=True)

# -------------------------------
# Sentiment by Region
# -------------------------------
st.subheader("Sentiment Distribution by Region")
region_df = df.groupby(["region", "sentiment_label"]).size().reset_index(name="count")
region_chart = alt.Chart(region_df).mark_bar().encode(
    x=alt.X("region:N", sort="-y"),
    y=alt.Y("count:Q"),
    color=alt.Color("sentiment_label:N",
                    scale=alt.Scale(domain=list(sentiment_colors.keys()),
                                    range=list(sentiment_colors.values())),
                    legend=alt.Legend(title="Sentiment")),
    tooltip=["region", "sentiment_label", "count"]
).interactive()
st.altair_chart(region_chart, use_container_width=True)

# -------------------------------
# Ratings vs Sentiment Scatter
# -------------------------------
st.subheader("Ratings vs Sentiment by Competitor Type")
scatter_chart = alt.Chart(df).mark_circle(size=80, opacity=0.8).encode(
    x=alt.X("rating:Q", title="Rating"),
    y=alt.Y("sentiment_score:Q", title="Sentiment Score"),
    color=alt.Color("sentiment_label:N",
                    scale=alt.Scale(domain=list(sentiment_colors.keys()),
                                    range=list(sentiment_colors.values())),
                    legend=alt.Legend(title="Sentiment")),
    tooltip=["brand", "industry", "region", "rating", "sentiment_label", "sentiment_score", "competitor_type"]
).interactive().properties(height=400)
st.altair_chart(scatter_chart, use_container_width=True)

# -------------------------------
# Top Competitors by Market Share
# -------------------------------
st.subheader("Top Competitors by Market Share")
top_competitors = df.groupby("brand")["market_share_est"].mean().reset_index()
top_competitors = top_competitors.sort_values("market_share_est", ascending=False).head(15)
bar_market_chart = alt.Chart(top_competitors).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
    x=alt.X("brand:N", sort="-y", title="Brand"),
    y=alt.Y("market_share_est:Q", title="Average Market Share (%)"),
    color=alt.value("#FF2E2E"),
    tooltip=["brand", "market_share_est"]
).properties(height=400)
st.altair_chart(bar_market_chart, use_container_width=True)