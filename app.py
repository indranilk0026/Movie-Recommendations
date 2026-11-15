import os
import random

import numpy as np
import pandas as pd
import streamlit as st

import features as ft
from recommender import DEFAULT_WEIGHTS, Recommender

st.set_page_config(page_title="CineMatch - Movie Recommender", page_icon="🎬", layout="wide")

ENGINES = {
    "Hybrid (blend of all signals)": "hybrid",
    "Semantic embeddings": "embed",
    "TF-IDF keywords": "tfidf",
    "Bag-of-Words (original baseline)": "count",
    "Collaborative (viewers' ratings)": "cf",
}
LANGS = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese",
         "zh": "Chinese", "ko": "Korean", "hi": "Hindi", "ru": "Russian", "pt": "Portuguese", "cn": "Cantonese",
         "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "ml": "Malayalam", "tr": "Turkish", "sv": "Swedish"}


# ---------------- load model ----------------
@st.cache_resource(show_spinner="Loading the recommendation models...")
def load_recommender():
    try:
        return Recommender.load("artifacts")
    except Exception:
        d = ft.find_data_dir()
        return Recommender.build_lite(d) if d else None


@st.cache_data
def load_eval():
    p = "artifacts/evaluation_results.csv"
    return pd.read_csv(p, index_col=0) if os.path.exists(p) else None


rec = load_recommender()
if rec is None:
    st.error("No model found. Add the `artifacts/` folder (from the notebook) or the two TMDB csv files "
             "(`tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`) in a `datasets/` folder.")
    st.stop()
meta = rec.meta


# ---------------- UI helpers ----------------
def card(row, q=None, show_why=True, extra=None):
    with st.container(border=True):
        year = f" ({row['year']})" if row["year"] else ""
        st.markdown(f"**{row['title']}**{year}")
        st.caption(f"⭐ {row['vote_average']:.1f}  ·  {', '.join(row['genres_l'][:3])}")
        if row["director"]:
            st.caption(f"🎬 {row['director']}  ·  {', '.join(row['cast_l'][:2])}")
        ov = row["overview"]
        st.write(ov[:200] + ("…" if len(ov) > 200 else ""))
        if extra:
            st.caption(extra)
        if show_why and q is not None:
            with st.expander("Why this movie?"):
                for r in rec.explain(q, int(row.name)) or ["Similar overall storyline and style"]:
                    st.write("• " + r)
                parts = [f"{lab}: {row['s_' + k]:.2f}" for k, lab in
                         [("tfidf", "keywords"), ("embed", "semantic"), ("cf", "viewers' taste")] if f"s_{k}" in row.index]
                st.caption("Similarity  |  " + "  |  ".join(parts))


def grid(df, q=None, per_row=3):
    if df.empty:
        st.info("No movies match these filters. Try relaxing them.")
        return
    for s in range(0, len(df), per_row):
        cols = st.columns(per_row)
        for col, (_, row) in zip(cols, df.iloc[s:s + per_row].iterrows()):
            with col:
                card(row, q, extra=f"Match score: {row['score']:.2f}")


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    if rec.mode == "lite":
        st.info("Lite mode: keyword (TF-IDF) engine. Add the `artifacts/` folder for embeddings, "
                "collaborative filtering and the hybrid model.")
    options = {k: v for k, v in ENGINES.items() if v == "hybrid" or v in rec.signals}
    method = options[st.selectbox("Recommendation engine", list(options), index=0)]
    n_recs = st.slider("Number of recommendations", 3, 12, 6)
    weights = dict(rec.weights)
    if method == "hybrid":
        with st.expander("Hybrid weights (advanced)"):
            for k, label in [("embed", "Semantic embeddings"), ("tfidf", "TF-IDF keywords"),
                             ("cf", "Viewers' ratings"), ("quality", "Rating / popularity prior")]:
                if k == "quality" or k in rec.signals:
                    weights[k] = st.slider(label, 0.0, 1.0, float(weights.get(k, DEFAULT_WEIGHTS[k])), 0.05)
    st.subheader("Filters")
    all_genres = sorted({g for gs in meta["genres_l"] for g in gs})
    langs = sorted(meta["original_language"].dropna().unique(), key=lambda c: (c not in LANGS, LANGS.get(c, c)))
    y_min, y_max = int(meta["year"].min()), int(meta["year"].max())
    genres = st.multiselect("Genres (any of)", all_genres)
    min_rating = st.slider("Minimum rating", 0.0, 9.0, 0.0, 0.5)
    language = st.selectbox("Original language", ["Any"] + langs, format_func=lambda c: LANGS.get(c, c))
    year_range = st.slider("Release year", y_min, y_max, (y_min, y_max))
filters = dict(genres=genres, min_rating=min_rating, language=language, year_range=year_range)

# ---------------- main ----------------
st.title("🎬 CineMatch: Movie Recommendation System")
st.caption("Content-based + collaborative + semantic recommendations on the TMDB 5000 movie dataset.")
tab_rec, tab_prof, tab_eda, tab_eval, tab_about = st.tabs(
    ["🎬 Recommend", "👤 My taste profile", "📊 Explore the data", "🧪 Model comparison", "ℹ️ About"])

# ---- tab 1: one movie
with tab_rec:
    st.session_state.setdefault("query", None)
    selected = st.selectbox("Choose a movie (type to search)", rec.labels, index=None, placeholder="Select a movie...")
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("🔎 Show recommendations"):
        if selected is None:
            st.info("Please choose a movie first.")
        else:
            st.session_state.query = rec.label_to_idx[selected]
    if c2.button("🎲 Surprise me"):
        st.session_state.query = random.randrange(rec.n)
    q = st.session_state.query
    if q is None:
        st.info("Choose a movie and press the button to get recommendations.")
    else:
        use = method
        if method == "cf" and not rec.has_cf[q]:
            st.warning("This movie has too few viewer ratings for the collaborative engine; using Hybrid instead.")
            use = "hybrid"
        st.subheader("Because you picked")
        left, _ = st.columns([1, 2])
        with left:
            card(meta.iloc[q], show_why=False)
        st.subheader("You might also like")
        grid(rec.recommend(q, n=n_recs, method=use, weights=weights, **filters), q)

# ---- tab 2: taste profile (several movies)
with tab_prof:
    st.write("Pick **2 to 10 movies you love**. The system builds a taste profile from all of them and finds movies "
             "that match the combination, which is useful when you have mixed tastes.")
    favs = st.multiselect("Your favourite movies", rec.labels, max_selections=10)
    if len(favs) < 2:
        st.info("Select at least two movies.")
    else:
        idxs = [rec.label_to_idx[f] for f in favs]
        use = "hybrid" if method == "cf" and not rec.has_cf[idxs].any() else method
        st.subheader("Recommended for your taste")
        grid(rec.recommend(idxs, n=n_recs, method=use, weights=weights, **filters), idxs)

# ---- tab 3: EDA
with tab_eda:
    st.subheader("Dataset overview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Movies", f"{len(meta):,}")
    m2.metric("Genres", len(all_genres))
    m3.metric("Directors", f"{meta['director'].nunique():,}")
    m4.metric("Average rating", f"{meta['vote_average'].mean():.2f}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Movies per genre**")
        st.bar_chart(meta.explode("genres_l")["genres_l"].value_counts().head(15))
        st.markdown("**Movies released per decade**")
        st.bar_chart((meta["year"] // 10 * 10).value_counts().sort_index())
    with b:
        st.markdown("**Distribution of ratings**")
        v = meta.loc[meta["vote_count"] >= 10, "vote_average"]
        hist, edges = np.histogram(v, bins=np.arange(0, 10.5, 0.5))
        st.bar_chart(pd.Series(hist, index=[f"{e:.1f}" for e in edges[:-1]]))
        st.markdown("**Top original languages**")
        st.bar_chart(meta["original_language"].map(lambda c: LANGS.get(c, c)).value_counts().head(8))
    st.markdown("**Highest-rated directors** (at least 3 movies, rating from movies with 50+ votes)")
    d = meta[(meta["vote_count"] >= 50) & (meta["director"] != "")].groupby("director")["vote_average"].agg(["mean", "count"])
    st.dataframe(d[d["count"] >= 3].sort_values("mean", ascending=False).head(10).round(2)
                 .rename(columns={"mean": "avg rating", "count": "movies"}))

# ---- tab 4: evaluation
with tab_eval:
    res = load_eval()
    st.subheader("How good are the recommendations?")
    st.write("Each method was tested on **unseen users**: for a movie a user liked, we asked for 10 recommendations "
             "and checked how many of the user's *other* liked movies were found (Precision@10) and whether at least "
             "one was found (Hit-rate@10). Hybrid weights were tuned on separate validation users.")
    if res is None:
        st.info("Run the notebook to generate `artifacts/evaluation_results.csv` and the results will appear here.")
    else:
        st.dataframe(res.round(4))
        st.bar_chart(res[["Precision@10", "HitRate@10"]])
        st.caption("Absolute numbers look small because each user liked only a few of thousands of candidate movies; "
                   "compare the methods with each other.")
    if rec.mode == "full":
        st.markdown("**Tuned hybrid weights**")
        st.bar_chart(pd.Series(rec.weights))

# ---- tab 5: about
with tab_about:
    st.markdown("""
### About this project
**CineMatch** recommends movies using several complementary signals:
1. **Bag-of-Words and TF-IDF**: compare the words in each movie's overview, genres, keywords, cast and director.
2. **Sentence embeddings** (`all-MiniLM-L6-v2`): understand meaning, so related stories match even without shared words.
3. **Collaborative filtering**: learns from MovieLens viewers' ratings (SVD) which movies are liked by the same people.
4. **Hybrid ranking**: blends the signals with a rating/popularity prior; weights are tuned on validation users.

**Extra features:** filters (genre, rating, language, year), taste profiles from several movies, explanations for every
recommendation, data exploration dashboard and an offline evaluation of all methods.

**Tech stack:** Python, pandas, NumPy, SciPy, scikit-learn, NLTK, sentence-transformers, Streamlit.

**Data:** TMDB 5000 Movie Dataset (movies and credits) and MovieLens (GroupLens) ratings.
MovieLens: F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS, 2015.
""")
