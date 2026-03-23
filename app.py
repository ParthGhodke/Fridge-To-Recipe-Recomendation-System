!pip install streamlit pyngrok kagglehub pandas scikit-learn
%%writefile app.py
import streamlit as st
import pandas as pd
import ast
import os
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    path = kagglehub.dataset_download("shuyangli94/food-com-recipes-and-user-interactions")
    file_path = os.path.join(path, "RAW_recipes.csv")

    df = pd.read_csv(file_path)
    df = df[['name', 'ingredients', 'steps']]

    df['ingredients'] = df['ingredients'].apply(ast.literal_eval)
    df['steps'] = df['steps'].apply(ast.literal_eval)

    df['ingredients'] = df['ingredients'].apply(lambda x: " ".join(x))
    df['steps'] = df['steps'].apply(lambda x: " ".join(x))

    df.columns = ['recipe_name', 'ingredients', 'instructions']
    df['ingredients'] = df['ingredients'].fillna('')

    df = df.sample(5000, random_state=42)
    return df

df = load_data()

@st.cache_resource
def create_model(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data['ingredients'])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_model(df)

def get_missing_ingredients(user_input, recipe_ingredients):
    return list(set(recipe_ingredients.split()) - set(user_input.split()))

def recommend(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, tfidf_matrix)
    idxs = similarity[0].argsort()[-5:][::-1]

    results = []
    for i in idxs:
        r = df.iloc[i]

        results.append({
            "name": r['recipe_name'],
            "ingredients": r['ingredients'],
            "instructions": r['instructions']
        })

    return results

st.title("🥗 Fridge → Recipe Recommender")

user_input = st.text_input("Enter ingredients")

if st.button("Recommend"):
    res = recommend(user_input)

    for recipe in res:
        st.write("## 🍽️", recipe['name'])
      

        st.write("**🧂 Ingredients:**")
        st.write(recipe['ingredients'])

        st.write("**👨‍🍳 Steps:**")
        st.write(recipe['instructions'][:500] + "...")

        st.write("---")

  from pyngrok import ngrok

# Replace with your ngrok token
ngrok.set_auth_token("3BKoeV0j51aFDrId98k01tZNQEJ_2CyvzHiRkHQuVSSpKadqe")

# Run Streamlit in background
!streamlit run app.py &>/content/logs.txt &

# Create public URL
public_url = ngrok.connect(8501)
public_url
