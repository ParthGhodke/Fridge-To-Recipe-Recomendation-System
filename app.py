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
st.write("Find recipes based on your ingredients!")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("recipes_small.csv")

    # Clean instructions
    def clean_steps(x):
        if isinstance(x, list):
            return x
        if isinstance(x, str) and "||" in x:
            return x.split("||")
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except:
                return [x]
        return []

    df['instructions'] = df['instructions'].apply(clean_steps)
    df = df.fillna('')
    
    # Reduce size
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
# Helper Functions
# -------------------------------

# Veg / Non-Veg classifier
def classify_veg(ingredients):
    nonveg_items = ["chicken", "egg", "fish", "meat", "mutton", "beef"]
    for item in nonveg_items:
        if item in ingredients.lower():
            return "Non-Veg"
    return "Veg"

# Meal type classifier
def classify_meal(recipe_name):
    name = recipe_name.lower()
    if any(x in name for x in ["breakfast", "pancake", "omelette", "toast"]):
        return "Breakfast"
    elif any(x in name for x in ["dinner", "curry", "rice", "biryani"]):
        return "Dinner"
    else:
        return "Lunch"

# Missing ingredients
def get_missing(user_input, recipe_ing):
    user_set = set(user_input.lower().split())
    recipe_set = set(recipe_ing.lower().split())
    return list(recipe_set - user_set)

# Recommendation
def recommend_recipes(user_input, top_n=10):
    user_vec = vectorizer.transform([user_input.lower()])
    sim = cosine_similarity(user_vec, tfidf_matrix)
    idxs = sim[0].argsort()[-top_n:][::-1]
    return df.iloc[idxs]

# -------------------------------
# UI Filters
# -------------------------------
user_input = st.text_input("🧂 Enter ingredients (e.g. tomato onion cheese)")

col1, col2 = st.columns(2)

with col1:
    veg_filter = st.selectbox("🥦 Veg / Non-Veg", ["All", "Veg", "Non-Veg"])

with col2:
    meal_filter = st.selectbox("🍽️ Meal Type", ["All", "Breakfast", "Lunch", "Dinner"])

# -------------------------------
# Show Results
# -------------------------------
if st.button("🔍 Recommend Recipes"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some ingredients!")

    else:
        results = recommend_recipes(user_input)

        st.success("✅ Top Recipes for You:")

        for _, row in results.iterrows():

            veg_type = classify_veg(row['ingredients'])
            meal_type = classify_meal(row['recipe_name'])

            # Apply filters
            if veg_filter != "All" and veg_type != veg_filter:
                continue

            if meal_filter != "All" and meal_type != meal_filter:
                continue

            st.subheader(f"🍽️ {row['recipe_name']}")

            st.write(f"🥦 Type: {veg_type} | 🍴 Meal: {meal_type}")

            st.markdown("**🧂 Ingredients:**")
            st.write(row['ingredients'])

            # Missing ingredients
            missing = get_missing(user_input, row['ingredients'])
            st.markdown("**🛒 You need to buy:**")
            if missing:
                st.write(", ".join(missing[:5]))
            else:
                st.write("Nothing! You have everything ✅")

            # Steps
            st.markdown("**👨‍🍳 Steps:**")
            for i, step in enumerate(row['instructions'], 1):
                st.write(f"{i}. {step}")

            st.markdown("---")
