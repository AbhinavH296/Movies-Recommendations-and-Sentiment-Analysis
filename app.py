import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64

# Load required files
with open('movies1.pkl', 'rb') as f:
    movies = pickle.load(f)

with open('tfidf_matrix_recommendation.pkl', 'rb') as f:
    tfidf_matrix_recommendation = pickle.load(f)

with open('tfidf_vectorizer_sentiment.pkl', 'rb') as f:
    vectorizer_sentiment = pickle.load(f)

with open('best_xgb_model.pkl', 'rb') as f:
    sentiment_model = pickle.load(f)

# Initialize NLTK components
nltk.download('stopwords')
nltk.download('vader_lexicon')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
analyzer = SentimentIntensityAnalyzer()

# Preprocess function for text with length-based handling
def preprocess_text_with_length_handling(text):
    """Preprocess text differently based on its length."""
    if len(text.split()) < 5:  # If short review
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
        text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
        return text.strip()  # Minimal preprocessing
    else:  # If long review
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        words = word_tokenize(text)
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        return TreebankWordDetokenizer().detokenize(words)

# Hybrid sentiment prediction function
def predict_sentiment_with_hybrid_approach(review, vectorizer, model):
    """Predict sentiment using hybrid approach."""
    if len(review.split()) < 5:  # If short review
        scores = analyzer.polarity_scores(review)
        if scores['compound'] > 0.1:
            return "Positive"
        elif scores['compound'] < -0.1:
            return "Negative"
        else:
            return "Neutral"
    else:  # For longer reviews, use the trained model
        clean_review = preprocess_text_with_length_handling(review)
        vectorized_review = vectorizer.transform([clean_review]).toarray()
        prediction = model.predict(vectorized_review)[0]
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return sentiment_map.get(prediction, "Neutral")

# Function to get movie recommendations
def get_recommendations(movie_title, data, tfidf_matrix, num_recommendations=6, similarity_threshold=0.2):
    from fuzzywuzzy import process

    best_match = process.extractOne(movie_title, data['Title'].values)
    if not best_match or best_match[1] < 80:
        return "Movie not found in the dataset."
    movie_title = best_match[0]
    movie_idx = data[data['Title'] == movie_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[movie_idx], tfidf_matrix)
    similar_indices = cosine_sim.argsort()[0][::-1][1:num_recommendations + 1]
    recommendations = []
    for idx in similar_indices:
        if cosine_sim[0][idx] >= similarity_threshold:
            recommendations.append(data.iloc[idx][['Title', 'Genre', 'Director', 'Cast', 'Description', 'Review', 'Poster']].to_dict())
    if not recommendations:
        return "No recommendations found with sufficient similarity."
    return recommendations

# Create Word Cloud background
def generate_wordcloud_image(data):
    text = " ".join(title for title in data['Title'])
    wordcloud = WordCloud(max_words=200, background_color="black", colormap='Blues').generate(text)
    wordcloud_path = "wordcloud.png"
    wordcloud.to_file(wordcloud_path)
    return wordcloud_path

# Function to set background and text styles
def set_background_and_styles(image_path):
    with open(image_path, "rb") as img_file:
        b64_image = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)),
                        url(data:image/png;base64,{b64_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white;
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: white;
        }}
        .stTextArea textarea {{
            background-color: rgba(255, 255, 255, 0.7);
            color: black;
        }}
        .stSelectbox div {{
            background-color: rgba(255, 255, 255, 0.7);
            color: black;
        }}
        .stButton>button {{
            background-color: #ff4b4b;
            color: white;
            border-radius: 12px;
        }}
        .stButton>button:hover {{
            background-color: #cc0000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Generate Word Cloud and set as background
wordcloud_path = generate_wordcloud_image(movies)
set_background_and_styles(wordcloud_path)

# Streamlit UI
st.title("Movie Recommendation and Sentiment Analysis")

# User Input
movie_list = movies['Title'].values
movie_title = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)
user_review = st.text_area("Enter your review for this movie:", "")

if st.button("Get Recommendations"):
    if not movie_title or not user_review:
        st.warning("Please enter both a movie title and your review!")
    else:
        # Predict sentiment for user's review
        user_sentiment = predict_sentiment_with_hybrid_approach(
        user_review, vectorizer_sentiment, sentiment_model
        )
        st.subheader(f"Your Review Sentiment: {user_sentiment}")

        # Get recommendations
        recommendations = get_recommendations(movie_title, movies, tfidf_matrix_recommendation)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.subheader(f"Recommendations based on {movie_title}:")

            # Display recommendations in rows
            cols = st.columns(3)  # 3 columns per row
            for i, rec in enumerate(recommendations):
                with cols[i % 3]:
                    st.image(rec['Poster'], width=200)
                    st.write(f"**Title**: {rec['Title']}")
                    st.write(f"**Genre**: {rec['Genre']}")
                    st.write(f"**Director**: {rec['Director']}")
                    st.write(f"**Cast**: {rec['Cast']}")
                    st.write(f"**Description**: {' '.join(rec['Description'])}")
                    rec_sentiment = predict_sentiment_with_hybrid_approach(
                        rec['Review'], vectorizer_sentiment, sentiment_model
                    )
                    st.write(f"**Review Sentiment**: {rec_sentiment}")
                    st.write("---")
