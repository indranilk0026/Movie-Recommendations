"""Data loading and feature engineering for the TMDB 5000 dataset.

Shared by the training notebook and by the Streamlit app (which can rebuild a lightweight
model straight from the CSV files if the full trained artifacts are not present).
"""
import ast
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

try:
    from nltk.stem.porter import PorterStemmer
    _stem = PorterStemmer().stem
except Exception:                      # nltk missing -> skip stemming instead of crashing
    _stem = lambda w: w

MOVIES_CSV, CREDITS_CSV = "tmdb_5000_movies.csv", "tmdb_5000_credits.csv"


def find_data_dir(candidates=("datasets", "data", ".", "..")):
    for d in candidates:
        if os.path.exists(os.path.join(d, MOVIES_CSV)) and os.path.exists(os.path.join(d, CREDITS_CSV)):
            return d
    return None


def parse(s):
    """The TMDB csv stores lists as JSON text, e.g. '[{"id": 28, "name": "Action"}]'."""
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return []


def load_tmdb(data_dir):
    """Merge movies + credits on the unique movie id and extract clean list columns."""
    movies = pd.read_csv(os.path.join(data_dir, MOVIES_CSV))
    credits = pd.read_csv(os.path.join(data_dir, CREDITS_CSV), engine="python")
    credits = credits.rename(columns={"movie_id": "id"}).drop(columns=["title"])
    df = movies.merge(credits, on="id", how="inner", validate="one_to_one")
    df["overview"] = df["overview"].fillna("")
    df = df.dropna(subset=["release_date"]).reset_index(drop=True)
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.astype(int)
    df["genres_l"] = df["genres"].apply(lambda s: [d["name"] for d in parse(s)])
    df["keywords_l"] = df["keywords"].apply(lambda s: [d["name"] for d in parse(s)])
    df["cast_l"] = df["cast"].apply(lambda s: [d["name"] for d in parse(s)][:3])
    df["director"] = df["crew"].apply(lambda s: next((d["name"] for d in parse(s) if d.get("job") == "Director"), ""))
    return df


def make_tags(r):
    """overview + genres + keywords + top-3 cast + director -> one stemmed lowercase string."""
    squash = lambda xs: [x.replace(" ", "") for x in xs]          # "Sam Worthington" -> "SamWorthington"
    toks = (r["overview"].split() + squash(r["genres_l"]) + squash(r["keywords_l"])
            + squash(r["cast_l"]) + squash([r["director"]] if r["director"] else []))
    return " ".join(_stem(t) for t in " ".join(toks).lower().split())


def rich_text(r):
    """Natural-language description used by the sentence-embedding model (no stemming)."""
    return (f"{r['title']}. Genres: {', '.join(r['genres_l'])}. {r['overview']} "
            f"Keywords: {', '.join(r['keywords_l'])}. Directed by {r['director']}. Starring {', '.join(r['cast_l'])}.")


def add_quality_and_labels(df):
    """IMDb-style weighted rating + log popularity -> 'quality' in 0..1, plus unique dropdown labels."""
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.70)
    v, R = df["vote_count"], df["vote_average"]
    df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
    mm = lambda s: (s - s.min()) / (s.max() - s.min())
    df["quality"] = 0.7 * mm(df["weighted_rating"]) + 0.3 * mm(np.log1p(df["popularity"]))
    df["label"] = df["title"] + " (" + df["year"].astype(str) + ")"
    dup = df["label"].duplicated(keep=False)
    df.loc[dup, "label"] = df.loc[dup, "label"] + " [id " + df.loc[dup, "id"].astype(str) + "]"
    return df


META_COLS = ["id", "title", "label", "year", "genres_l", "keywords_l", "cast_l", "director", "overview",
             "vote_average", "vote_count", "popularity", "original_language", "quality"]


def build_features(df):
    """Adds tags/text/quality and returns (df, meta, count_matrix, tfidf_matrix). Rows are L2-normalised."""
    df = df.copy()
    df["tags"] = df.apply(make_tags, axis=1)
    df["text"] = df.apply(rich_text, axis=1)
    df = add_quality_and_labels(df)
    X_count = sp.csr_matrix(normalize(CountVectorizer(max_features=5000, stop_words="english").fit_transform(df["tags"])))
    X_tfidf = sp.csr_matrix(normalize(TfidfVectorizer(max_features=5000, stop_words="english",
                                                      sublinear_tf=True).fit_transform(df["tags"])))
    return df, df[META_COLS].copy(), X_count, X_tfidf
