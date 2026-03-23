import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🥗 Fridge → Recipe Recommender")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_cleaned.csv")
    df = df.fillna('')
    return df.sample(5000, random_state=42)

df = load_data()

# Model
@st.cache_resource
def create_model(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['ingredients'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_model(df)

# Recommend function
def recommend(user_input):
    user_vec = vectorizer.transform([user_input])
    sim = cosine_similarity(user_vec, tfidf_matrix)
    idxs = sim[0].argsort()[-5:][::-1]
    return df.iloc[idxs]

# UI
user_input = st.text_input("Enter ingredients")

if st.button("Recommend"):
    results = recommend(user_input)
    
    for _, r in results.iterrows():
        st.subheader(r['recipe_name'])
        st.write("🧂 Ingredients:", r['ingredients'])
        
        st.write("👨‍🍳 Steps:")
        for step in eval(r['instructions']):
            st.write("•", step)
        
        st.markdown("---")


# Run Streamlit in background
!streamlit run app.py &>/content/logs.txt &

# Create public URL
public_url = ngrok.connect(8501)
public_url
