
import os
import streamlit as st
import pandas as pd

from recommender import Recommender


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #E50914;
        text-align: center;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 16px;
        margin-bottom: 30px;
    }

    .movie-title {
        font-size: 19px;
        font-weight: 700;
        color: #ffffff;
    }

    .movie-info {
        color: #bbbbbb;
        font-size: 14px;
    }

    div.stButton > button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        width: 100%;
    }

    div.stButton > button:hover {
        background-color: #b20710;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading CineMatch recommendation model...")
def load_recommender():
    return Recommender.load(ARTIFACTS_DIR)


try:
    rec = load_recommender()

except Exception as e:
    st.error("Unable to load the CineMatch model.")

    st.markdown(
        """
        Please check that your GitHub repository contains:

        ```text
        artifacts/
            meta.csv.gz
            count.npz
            tfidf.npz
            weights.json
        ```

        Also verify that `recommender.py` is the updated version.
        """
    )

    with st.expander("Technical error"):
        st.exception(e)

    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 CineMatch</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Discover movies that match your taste.'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"Recommendation engine: {rec.mode.upper()} "
    f"| Movies available: {rec.n:,}"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🎥 CineMatch")
st.sidebar.markdown("---")

st.sidebar.subheader("Recommendation Settings")

method_options = {
    "Hybrid Recommendation": "hybrid",
    "TF-IDF Similarity": "tfidf",
    "Bag of Words Similarity": "count",
    "Movie Quality": "quality",
}

selected_method = st.sidebar.selectbox(
    "Recommendation Method",
    list(method_options.keys()),
)

method = method_options[selected_method]

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

all_genres = sorted({
    genre
    for genres in rec.meta["genres_l"]
    for genre in genres
})

selected_genres = st.sidebar.multiselect(
    "Genres",
    all_genres,
)

min_rating = st.sidebar.slider(
    "Minimum TMDB Rating",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
)

languages = sorted(
    rec.meta["original_language"]
    .dropna()
    .unique()
    .tolist()
)

selected_language = st.sidebar.selectbox(
    "Original Language",
    ["Any"] + languages,
)

year_min = int(rec.meta["year"].min())
year_max = int(rec.meta["year"].max())

year_range = st.sidebar.slider(
    "Release Year",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

number_of_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
)


# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------

tab1, tab2 = st.tabs(
    ["🎬 Movie Recommendations", "📊 About CineMatch"]
)


# --------------------------------------------------
# TAB 1: RECOMMENDATIONS
# --------------------------------------------------

with tab1:

    st.subheader("Find Your Next Movie")

    movie_options = rec.meta["label"].tolist()

    selected_movie = st.selectbox(
        "Search and select a movie",
        movie_options,
        index=None,
        placeholder="Type a movie title...",
    )

    recommend_button = st.button(
        "✨ Find Similar Movies",
        use_container_width=True,
    )

    if recommend_button:

        if not selected_movie:
            st.warning("Please select a movie first.")

        else:

            try:
                with st.spinner("Finding movies you may enjoy..."):

                    recommendations = rec.recommend(
                        selected_movie,
                        n=number_of_recommendations,
                        method=method,
                        genres=selected_genres or None,
                        min_rating=min_rating,
                        language=selected_language,
                        year_range=year_range,
                    )

                if recommendations.empty:
                    st.warning(
                        "No movies match the selected filters. "
                        "Try relaxing your filters."
                    )

                else:

                    st.success(
                        f"Found {len(recommendations)} recommendations!"
                    )

                    st.markdown(
                        f"### Because you selected: **{selected_movie}**"
                    )

                    for i, (_, movie) in enumerate(
                        recommendations.iterrows(),
                        start=1,
                    ):

                        with st.container(border=True):

                            col1, col2 = st.columns(
                                [0.12, 0.88]
                            )

                            with col1:
                                st.markdown(f"## {i}")

                            with col2:

                                st.markdown(
                                    f"### {movie['label']}"
                                )

                                details = []

                                if pd.notna(movie.get("year")):
                                    details.append(
                                        str(int(movie["year"]))
                                    )

                                if pd.notna(
                                    movie.get("vote_average")
                                ):
                                    details.append(
                                        f"⭐ {movie['vote_average']:.1f}/10"
                                    )

                                if movie.get("genres_l"):
                                    details.append(
                                        ", ".join(movie["genres_l"])
                                    )

                                st.caption(" | ".join(details))

                                if movie.get("overview"):
                                    st.write(movie["overview"])

                                movie_index = int(movie.name)

                                reasons = rec.explain(
                                    rec.label_to_idx[selected_movie],
                                    movie_index,
                                )

                                if reasons:
                                    with st.expander(
                                        "Why this recommendation?"
                                    ):
                                        for reason in reasons:
                                            st.write(f"• {reason}")

                                if "score" in movie:
                                    st.caption(
                                        f"Recommendation score: "
                                        f"{movie['score']:.4f}"
                                    )

            except Exception as e:
                st.error(
                    "Something went wrong while generating recommendations."
                )

                with st.expander("Technical error"):
                    st.exception(e)


# --------------------------------------------------
# TAB 2: ABOUT
# --------------------------------------------------

with tab2:

    st.subheader("About CineMatch")

    st.write(
        """
        CineMatch is a content-based movie recommendation system
        designed to help users discover movies based on their
        preferences and movie characteristics.
        """
    )

    st.markdown("### Recommendation Techniques")

    st.markdown(
        """
        **1. TF-IDF (Term Frequency–Inverse Document Frequency)**

        Identifies similarities between movies using their textual
        information, such as genres, keywords, cast and descriptions.

        **2. Bag of Words (BoW)**

        Represents movie information through word-frequency features.

        **3. Quality-Based Recommendation**

        Uses movie rating information as an additional recommendation signal.

        **4. Hybrid Recommendation**

        Combines TF-IDF similarity and movie quality scores to generate
        recommendations.
        """
    )

    st.markdown("### Dataset")

    st.write(
        """
        CineMatch uses the TMDB 5000 Movies and Credits datasets.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Movies", f"{rec.n:,}")

    col2.metric(
        "Recommendation Mode",
        rec.mode.upper(),
    )

    col3.metric(
        "Available Signals",
        str(len(rec.signals)),
    )

    st.markdown("---")

    st.caption(
        "CineMatch | Movie Recommendation System"
    )
