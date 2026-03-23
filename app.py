import streamlit as st
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Fridge to Recipe", layout="centered")

st.title("🥗 Fridge → Recipe Recommender")
st.write("Turn your available ingredients into delicious recipes!")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_cleaned.csv")
    
    # Convert string list back to list
    df['instructions'] = df['instructions'].apply(ast.literal_eval)
    
    df = df.fillna('')
    
    # Reduce size for faster performance
    df = df.sample(5000, random_state=42)
    
    return df

df = load_data()

# -------------------------------
# Create Model
# -------------------------------
@st.cache_resource
def create_model(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['ingredients'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_model(df)

# -------------------------------
# Recommendation Function
# -------------------------------
def recommend_recipes(user_input, top_n=5):
    user_input = user_input.lower()
    
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    
    return df.iloc[top_indices]

# -------------------------------
# UI Input
# -------------------------------
user_input = st.text_input("🧂 Enter ingredients (e.g. tomato onion cheese)")

# -------------------------------
# Show Results
# -------------------------------
if st.button("🔍 Recommend Recipes"):
    
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some ingredients!")
    
    else:
        results = recommend_recipes(user_input)
        
        st.success("✅ Top Recipes for You:")
        
        for index, row in results.iterrows():
            
            st.subheader(f"🍽️ {row['recipe_name']}")
            
            st.markdown("**🧂 Ingredients:**")
            st.write(row['ingredients'])
            
            st.markdown("**👨‍🍳 Steps:**")
            for step in row['instructions']:
                st.write("•", step)
            
            st.markdown("---")
