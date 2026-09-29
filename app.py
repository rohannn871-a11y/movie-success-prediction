import streamlit as st
import joblib
import pandas as pd

# ============================================================
# LOAD MODEL
# ============================================================

model_data = joblib.load("movie_success_model_final.pkl")

model = model_data["model"]
scaler = model_data["scaler"]
label_encoders = model_data["label_encoders"]
features = model_data["features"]


# ============================================================
# TITLE
# ============================================================

st.title("🎬 Movie Success Prediction")

st.write(
    "Predict whether a movie is likely to be a Flop, Average, or Hit "
    "using a Machine Learning model."
)


# ============================================================
# MOVIE INFORMATION
# ============================================================

st.header("🎥 Movie Information")

st.write("Enter the basic information about the movie.")

col1, col2 = st.columns(2)

with col1:

    color = st.selectbox(
        "Color",
        label_encoders["color"].classes_
    )

    duration = st.number_input(
        "Duration (minutes)",
        min_value=1,
        value=120,
        step=1
    )

    title_year = st.number_input(
        "Release Year",
        min_value=1900,
        max_value=2026,
        value=2015,
        step=1
    )

    language = st.selectbox(
        "Language",
        label_encoders["language"].classes_
    )

    country = st.selectbox(
        "Country",
        label_encoders["country"].classes_
    )


with col2:

    genres = st.selectbox(
        "Genres",
        label_encoders["genres"].classes_
    )

    content_rating = st.selectbox(
        "Content Rating",
        label_encoders["content_rating"].classes_
    )

    aspect_ratio = st.number_input(
        "Aspect Ratio",
        min_value=0.1,
        value=1.85,
        step=0.01,
        format="%.2f"
    )

    facenumber_in_poster = st.number_input(
        "Number of Faces in Poster",
        min_value=0,
        value=1,
        step=1
    )


# ============================================================
# CAST & POPULARITY
# ============================================================

st.header("🎭 Cast & Popularity")

col1, col2 = st.columns(2)

with col1:

    actor_1_facebook_likes = st.number_input(
        "Actor 1 Facebook Likes",
        min_value=0,
        value=100,
        step=1
    )

    actor_2_facebook_likes = st.number_input(
        "Actor 2 Facebook Likes",
        min_value=0,
        value=100,
        step=1
    )

    actor_3_facebook_likes = st.number_input(
        "Actor 3 Facebook Likes",
        min_value=0,
        value=100,
        step=1
    )

    director_facebook_likes = st.number_input(
        "Director Facebook Likes",
        min_value=0,
        value=100,
        step=1
    )

    cast_total_facebook_likes = st.number_input(
        "Cast Total Facebook Likes",
        min_value=0,
        value=1000,
        step=100
    )


with col2:

    movie_facebook_likes = st.number_input(
        "Movie Facebook Likes",
        min_value=0,
        value=1000,
        step=100
    )

    num_voted_users = st.number_input(
        "Number of Voted Users",
        min_value=0,
        value=10000,
        step=100
    )

    num_critic_for_reviews = st.number_input(
        "Number of Critic Reviews",
        min_value=0,
        value=100,
        step=1
    )

    num_user_for_reviews = st.number_input(
        "Number of User Reviews",
        min_value=0,
        value=100,
        step=1
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.header("💰 Financial Information")

col1, col2 = st.columns(2)

with col1:

    budget = st.number_input(
        "Budget",
        min_value=0,
        value=1000000,
        step=1000
    )


with col2:

    gross = st.number_input(
        "Gross Revenue",
        min_value=0,
        value=1000000,
        step=1000
    )


# ============================================================
# PREPARE INPUT DATA
# ============================================================

input_data = pd.DataFrame([{

    "color": label_encoders["color"].transform([color])[0],

    "num_critic_for_reviews": num_critic_for_reviews,

    "duration": duration,

    "director_facebook_likes": director_facebook_likes,

    "actor_3_facebook_likes": actor_3_facebook_likes,

    "actor_1_facebook_likes": actor_1_facebook_likes,

    "gross": gross,

    "genres": label_encoders["genres"].transform([genres])[0],

    "num_voted_users": num_voted_users,

    "cast_total_facebook_likes": cast_total_facebook_likes,

    "facenumber_in_poster": facenumber_in_poster,

    "num_user_for_reviews": num_user_for_reviews,

    "language": label_encoders["language"].transform([language])[0],

    "country": label_encoders["country"].transform([country])[0],

    "content_rating": label_encoders["content_rating"].transform([content_rating])[0],

    "budget": budget,

    "title_year": title_year,

    "actor_2_facebook_likes": actor_2_facebook_likes,

    "aspect_ratio": aspect_ratio,

    "movie_facebook_likes": movie_facebook_likes

}])


# Arrange columns in the same order used during model training

input_data = input_data[features]


# ============================================================
# PREDICTION
# ============================================================

st.header("🔮 Prediction")


if st.button("Predict Movie Success"):

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Convert encoded prediction back to category name
    prediction_label = label_encoders["Classify"].inverse_transform(
        [prediction]
    )[0]


    # ========================================================
    # CATEGORY-WISE OUTPUT
    # ========================================================

    if prediction_label == "Hit":

        st.success(
            f"Predicted Movie Category: {prediction_label}"
        )

    elif prediction_label == "Average":

        st.warning(
            f"Predicted Movie Category: {prediction_label}"
        )

    elif prediction_label == "Flop":

        st.error(
            f"Predicted Movie Category: {prediction_label}"
        )


    # ========================================================
    # PREDICTION PROBABILITIES
    # ========================================================

    probabilities = model.predict_proba(input_scaled)[0]

    class_names = label_encoders["Classify"].inverse_transform(
        model.classes_
    )


    probability_df = pd.DataFrame({

        "Category": class_names,

        "Probability": probabilities

    })


    probability_df["Probability"] = (
        probability_df["Probability"] * 100
    ).round(0).astype(int)


    probability_df["Probability"] = (
        probability_df["Probability"].astype(str) + "%"
    )


    st.subheader("Prediction Probabilities")


    st.dataframe(
        probability_df,
        hide_index=True,
        use_container_width=True
    )