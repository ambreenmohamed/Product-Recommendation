import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK data is downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Function for sentiment analysis
def sentiment_analysis(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Function to load and preprocess data
@st.cache_data
def load_data():
    # Replace with your data loading method
    df = pd.read_csv('amazon.csv')  # Ensure your data is in CSV format
    df = df.dropna().drop_duplicates()
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'].astype(str).str.replace('|', ''), errors='coerce')
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(int)
    df['product_name'] = df['product_name'].apply(clean_text)
    df['about_product'] = df['about_product'].apply(clean_text)
    df['review_content'] = df['review_content'].apply(clean_text)
    df['combined_text'] = df['product_name'] + ' ' + df['category'] + ' ' + df['about_product'] + ' ' + df['review_content']
    df['combined_text'] = df['combined_text'].fillna('')
    df['Sentiment'] = df['review_content'].apply(sentiment_analysis)
    return df

# Function to compute TF-IDF matrix and cosine similarity
@st.cache_data
def compute_similarity(df):
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, ngram_range=(1, 1))
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    cosine_sim = cosine_similarity(tfidf_matrix)
    return cosine_sim

# Function to recommend products
def hybrid_recommendation(product_id, cosine_sim, df, top_n=10):
    idx = df.index[df['product_id'] == product_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    content_recommendations_idx = [i[0] for i in sim_scores[1:top_n+1]]
    recommended_products = df.iloc[content_recommendations_idx][['product_id', 'product_name', 'rating']]
    return recommended_products

# Streamlit app with custom styling
def main():
    # Custom CSS for styling
    st.markdown("""
        <style>
        body {
            background-color: #f6f1f1;
        }
        .title {
            color: #42153c;
            font-size: 40px;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #42153c;
            color: white;
            font-weight: bold;
        }
        .stSelectbox select {
            background-color: #e0cfe9;
            border: 1px solid #42153c;
            color: #42153c;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Product Recommendation System", anchor="product_recommendation_system")
    df = load_data()
    cosine_sim = compute_similarity(df)

    product_list = df['product_name'].unique()
    selected_product = st.selectbox("Select a Product", product_list)

    if st.button("Recommend"):
        product_id = df[df['product_name'] == selected_product]['product_id'].values[0]
        recommendations = hybrid_recommendation(product_id, cosine_sim, df)
        st.write("Recommendations for:", selected_product)
        for idx, row in recommendations.iterrows():
            st.write(f"Product Name: {row['product_name']}, Rating: {row['rating']}")

if __name__ == "__main__":
    main()
