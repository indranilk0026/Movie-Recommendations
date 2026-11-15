# 🎬 CineMatch: Movie Recommendation System

A multi-signal movie recommender built on the TMDB 5000 dataset and MovieLens ratings, deployed with Streamlit.

## Features
- **Five engines:** Bag-of-Words (baseline), TF-IDF, sentence embeddings (all-MiniLM-L6-v2), collaborative filtering (SVD on MovieLens ratings) and a tuned **hybrid** model.
- **Taste profile:** choose several favourite movies and get recommendations for the combination.
- **Explanations:** every recommendation shows why (shared genres, director, cast, themes).
- **Filters:** genre, minimum rating, original language, release year.
- **Data dashboard** and **model comparison** pages (offline evaluation on unseen users).
- Runs **without any API key**. If `artifacts/` is missing, the app rebuilds a lightweight TF-IDF model from the CSV files.

## Project structure
```
app.py                 Streamlit user interface
features.py            data loading and feature engineering
recommender.py         recommendation engine (shared by the notebook and the app)
CineMatch_Notebook.ipynb   full pipeline: EDA, models, evaluation, report generation
artifacts/             trained model files created by the notebook
datasets/              tmdb_5000_movies.csv, tmdb_5000_credits.csv
requirements.txt
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the models
Open `CineMatch_Notebook.ipynb` (Google Colab works well), upload the two TMDB csv files to `datasets/`, and run all cells.
It creates `artifacts/` and the project report.

## Data and credits
- TMDB 5000 Movie Dataset (movies and credits).
- MovieLens (GroupLens): F. M. Harper and J. A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS, 2015.
- Sentence-BERT: N. Reimers and I. Gurevych, EMNLP 2019.
