"""Recommendation engine shared by the notebook (evaluation) and the Streamlit app.

Signals (each is optional except count/tfidf):
  count  - Bag-of-Words cosine similarity            (original baseline)
  tfidf  - TF-IDF cosine similarity
  embed  - sentence-embedding cosine similarity      (needs artifacts from the notebook)
  cf     - collaborative filtering (MovieLens SVD)   (needs artifacts from the notebook)
  quality- popularity/rating prior
  hybrid - weighted blend of the signals above
Similarities are computed on the fly for the query movie(s), so no N x N matrix is stored.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

DEFAULT_WEIGHTS = {"tfidf": 0.25, "embed": 0.35, "cf": 0.30, "quality": 0.10}
LITE_WEIGHTS = {"tfidf": 0.85, "quality": 0.15}
LIST_COLS = ["genres_l", "keywords_l", "cast_l"]


def _minmax(v, mask):
    """Scale scores to 0-1 using only the candidate movies (mask == True)."""
    if not mask.any():
        return np.zeros_like(v, dtype=float)
    x = np.where(mask, v, np.nan)
    lo, hi = np.nanmin(x), np.nanmax(x)
    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v, dtype=float)
    return np.where(mask, out, 0.0)


def top_n(scores, n):
    """Indices of the n best scores (descending); -inf entries are dropped."""
    n = max(1, min(n, len(scores) - 1))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx[np.isfinite(scores[idx])]


def _as_list(q):
    return [int(x) for x in np.atleast_1d(q)]


class Recommender:
    def __init__(self, meta, count, tfidf, emb=None, cf=None, has_cf=None, weights=None, mode="full"):
        self.meta = meta.reset_index(drop=True)
        self.n = len(self.meta)
        self.count = sp.csr_matrix(count)
        self.tfidf = sp.csr_matrix(tfidf)
        self.emb = None if emb is None else np.asarray(emb, dtype=np.float32)
        self.cf = None if cf is None else np.asarray(cf, dtype=np.float32)
        self.has_cf = np.zeros(self.n, bool) if has_cf is None else np.asarray(has_cf, dtype=bool)
        self.mode = mode
        full = self.emb is not None and self.cf is not None
        self.weights = dict(weights or (DEFAULT_WEIGHTS if full else LITE_WEIGHTS))
        self.signals = ["count", "tfidf"] + (["embed"] if self.emb is not None else []) + (["cf"] if self.cf is not None else [])
        self.quality = self.meta["quality"].to_numpy(float)
        self.labels = self.meta["label"].tolist()
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    # ---------- save / load / lite build ----------
    def save(self, art_dir="artifacts"):
        os.makedirs(art_dir, exist_ok=True)
        m = self.meta.copy()
        for c in LIST_COLS:                      # lists -> "a|b|c" (robust across pandas versions)
            m[c] = m[c].apply(lambda xs: "|".join(xs))
        m.to_csv(f"{art_dir}/meta.csv.gz", index=False)
        sp.save_npz(f"{art_dir}/count.npz", self.count)
        sp.save_npz(f"{art_dir}/tfidf.npz", self.tfidf)
        np.save(f"{art_dir}/emb.npy", self.emb)
        np.save(f"{art_dir}/cf.npy", self.cf)
        np.save(f"{art_dir}/has_cf.npy", self.has_cf)
        with open(f"{art_dir}/weights.json", "w") as f:
            json.dump(self.weights, f, indent=2)

    @staticmethod
    def _read_meta(path):
        meta = pd.read_csv(path)
        for c in LIST_COLS:
            meta[c] = meta[c].fillna("").apply(lambda s: [x for x in s.split("|") if x])
        meta["overview"] = meta["overview"].fillna("")
        meta["director"] = meta["director"].fillna("")
        return meta

    @classmethod
    def load(cls, art_dir="artifacts"):
        weights = None
        if os.path.exists(f"{art_dir}/weights.json"):
            with open(f"{art_dir}/weights.json") as f:
                weights = json.load(f)
        return cls(cls._read_meta(f"{art_dir}/meta.csv.gz"), sp.load_npz(f"{art_dir}/count.npz"),
                   sp.load_npz(f"{art_dir}/tfidf.npz"), np.load(f"{art_dir}/emb.npy"), np.load(f"{art_dir}/cf.npy"),
                   np.load(f"{art_dir}/has_cf.npy"), weights, mode="full")

    @classmethod
    def build_lite(cls, data_dir):
        """TF-IDF + quality model built straight from the two TMDB csv files (no notebook needed)."""
        import features as ft
        df = ft.load_tmdb(data_dir)
        df, meta, X_count, X_tfidf = ft.build_features(df)
        return cls(meta, X_count, X_tfidf, mode="lite")

    # ---------- scoring ----------
    def components(self, q):
        """Raw similarity of every movie to the query movie(s) for each available signal.
        q can be one index or a list of indices (a 'taste profile': similarity to the average of the movies)."""
        idxs = _as_list(q)
        out = {}
        for name, M in (("count", self.count), ("tfidf", self.tfidf)):
            out[name] = np.asarray(M @ np.asarray(M[idxs].mean(axis=0)).ravel()).ravel()
        if self.emb is not None:
            out["embed"] = self.emb @ self.emb[idxs].mean(axis=0)
        if self.cf is not None:
            ok = [i for i in idxs if self.has_cf[i]]
            out["cf"] = self.cf @ self.cf[ok].mean(axis=0) if ok else np.zeros(self.n, dtype=np.float32)
        return out

    def filter_mask(self, genres=None, min_rating=0.0, language=None, year_range=None, pool=None):
        m = np.ones(self.n, dtype=bool) if pool is None else pool.copy()
        if genres:
            wanted = set(genres)
            m &= self.meta["genres_l"].apply(lambda g: bool(wanted.intersection(g))).to_numpy()
        if min_rating:
            m &= (self.meta["vote_average"] >= min_rating).to_numpy()
        if language and language != "Any":
            m &= (self.meta["original_language"] == language).to_numpy()
        if year_range:
            m &= self.meta["year"].between(year_range[0], year_range[1]).to_numpy()
        return m

    def score(self, q, method="hybrid", weights=None, mask=None, comps=None):
        idxs = _as_list(q)
        mask = np.ones(self.n, dtype=bool) if mask is None else mask.copy()
        mask[idxs] = False                               # never recommend the query movie(s)
        if method == "quality":
            return np.where(mask, self.quality, -np.inf)
        comps = comps or self.components(idxs)
        if method != "hybrid":
            return np.where(mask, comps[method], -np.inf)
        w = {k: v for k, v in dict(weights or self.weights).items() if k == "quality" or k in comps}
        if "cf" in w and not self.has_cf[idxs].any():    # query has no rating data -> drop the CF signal
            w["cf"] = 0.0
        total = sum(w.values())
        if total <= 0:
            return np.where(mask, self.quality, -np.inf)
        s = np.zeros(self.n)
        for k in ("tfidf", "embed", "cf"):
            if w.get(k, 0) > 0:
                v = np.clip(comps[k], 0, None) if k == "cf" else comps[k]
                s += (w[k] / total) * _minmax(v, mask)
        s += (w.get("quality", 0) / total) * self.quality
        return np.where(mask, s, -np.inf)

    def recommend(self, movie, n=10, method="hybrid", weights=None, **filters):
        """movie = label, index, or a list of them (taste profile). filters: genres, min_rating, language, year_range."""
        items = movie if isinstance(movie, (list, tuple, np.ndarray)) else [movie]
        idxs = [self.label_to_idx[m] if isinstance(m, str) else int(m) for m in items]
        comps = self.components(idxs)
        mask = self.filter_mask(**filters)
        s = self.score(idxs, method, weights, mask, comps)
        top = top_n(s, n)
        out = self.meta.iloc[top].copy()
        out["score"] = s[top]
        for k, v in comps.items():
            out[f"s_{k}"] = v[top]
        return out

    # ---------- explanations ----------
    def explain(self, q, j):
        """Human-readable reasons why movie j is similar to the query movie(s)."""
        idxs = _as_list(q)
        m = self.meta
        g, k, c = set(), set(), set()
        directors = {m.at[i, "director"] for i in idxs if m.at[i, "director"]}
        for i in idxs:
            g |= set(m.at[i, "genres_l"]); k |= set(m.at[i, "keywords_l"]); c |= set(m.at[i, "cast_l"])
        reasons = []
        sg = [x for x in m.at[j, "genres_l"] if x in g]
        if sg: reasons.append("Shared genres: " + ", ".join(sg))
        if m.at[j, "director"] and m.at[j, "director"] in directors: reasons.append(f"Same director: {m.at[j, 'director']}")
        sc = [x for x in m.at[j, "cast_l"] if x in c]
        if sc: reasons.append("Shared cast: " + ", ".join(sc))
        sk = [x for x in m.at[j, "keywords_l"] if x in k][:5]
        if sk: reasons.append("Shared themes: " + ", ".join(sk))
        return reasons
