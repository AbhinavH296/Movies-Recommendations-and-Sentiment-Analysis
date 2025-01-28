# Movie Recommendation and Sentiment Analysis

This project combines a **movie recommendation system** with a **sentiment analysis tool**, allowing users to discover new movies based on their preferences and analyze their reviews. Below is a detailed guide to the implementation and functionality.

## Features

1. **Movie Recommendation System**
   - Uses **TF-IDF vectorization** to process movie features (Title, Genre, Director, Cast, Description).
   - Employs **cosine similarity** to find and suggest movies similar to the selected title.

2. **Sentiment Analysis**
   - Predicts sentiment of user reviews using a **hybrid approach**:
     - **Short Reviews**: Analyzed with VADER (lexicon-based sentiment analysis).
     - **Long Reviews**: Preprocessed and evaluated using a trained **XGBoost model**.

3. **Interactive User Interface**
   - Built with **Streamlit**.
   - Includes features like movie dropdown selection, review input, and visual recommendations with sentiment analysis.

4. **Custom Styling**
   - Dynamic **word cloud** background for the UI.
   - Styled buttons, text boxes, and layouts for a user-friendly experience.

---

## Setup Instructions

### 1. Prerequisites
Ensure the following are installed:
- Python 3.8+
- Required libraries (install via `pip install -r requirements.txt`):
  ```plaintext
  streamlit
  pandas
  pickle
  sklearn
  nltk
  matplotlib
  wordcloud
  fuzzywuzzy
  xgboost
  base64
  ```

### 2. Files Required
The following files are essential for the application to run:
- `movies1.pkl`: Contains movie data (Title, Genre, Director, Cast, Description, etc.).
- `tfidf_matrix_recommendation.pkl`: Precomputed TF-IDF matrix for movie recommendations.
- `tfidf_vectorizer_sentiment.pkl`: Vectorizer for sentiment analysis.
- `best_xgb_model.pkl`: Trained XGBoost model for sentiment analysis.

### 3. NLTK Setup
Run these commands to download necessary NLTK components:
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')
```

### 4. Run the Application
Launch the app using Streamlit:
```bash
streamlit run app.py
```

---

## Functionality

### **1. Preprocessing User Reviews**
- Short reviews: Minimal preprocessing (lowercase, remove URLs and symbols).
- Long reviews: Tokenized, stopwords removed, lemmatized, and detokenized.

### **2. Hybrid Sentiment Prediction**
- Short reviews: Sentiment calculated using VADER.
- Long reviews: Sentiment predicted using the XGBoost model with the following classes:
  - **Negative** (0)
  - **Neutral** (1)
  - **Positive** (2)

### **3. Movie Recommendations**
- **Fuzzy Matching**: Matches user-input movie title to dataset.
- **Similarity Scoring**: Finds similar movies based on cosine similarity and a similarity threshold.

### **4. Word Cloud**
- Dynamically generated using all movie titles in the dataset.
- Set as the background of the Streamlit application.

---

## How to Use

1. Select a movie from the dropdown or type the title.
2. Enter your review for the selected movie.
3. Click **Get Recommendations**.
4. View:
   - Sentiment analysis of your review.
   - Recommended movies with details (poster, genre, director, cast, description).
   - Sentiment of reviews for recommended movies.

---

## Example Workflow

1. Select the movie **"Inception"**.
2. Enter your review: _"An absolute masterpiece with brilliant storytelling."_
3. Get output:
   - **Your Review Sentiment**: Positive
   - **Recommended Movies**:
     - **Title**: Interstellar
     - **Genre**: Sci-Fi
     - **Review Sentiment**: Positive

---

## Additional Notes
- Recommendations are filtered by a similarity threshold to ensure relevance.
- All processing is optimized for both short and long reviews.
- Background and UI styling enhance user experience.
