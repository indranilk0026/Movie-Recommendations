"""CineMatch recommendation engine: Lite and Full model support."""

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_WEIGHTS = {
    "tfidf": 0.25,
    "embed": 0.35,
    "cf": 0.30,
    "quality": 0.10,
}

LITE_WEIGHTS = {
    "tfidf": 0.85,
    "quality": 0.15,
}

LIST_COLS = ["genres_l", "keywords_l", "cast_l"]


def _minmax(v, mask):
    """Scale scores to 0-1 using candidate movies only."""
    if not mask.any():
        return np.zeros_like(v, dtype=float)

    x = np.where(mask, v, np.nan)
    lo, hi = np.nanmin(x), np.nanmax(x)

    out = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v, dtype=float)
    return np.where(mask, out, 0.0)


def top_n(scores, n):
    """Return indices of the n highest finite scores."""
    n = max(1, min(n, len(scores) - 1))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx[np.isfinite(scores[idx])]


def _as_list(q):
    return [int(x) for x in np.atleast_1d(q)]


class Recommender:

    def __init__(
        self,
        meta,
        count,
        tfidf,
        emb=None,
        cf=None,
        has_cf=None,
        weights=None,
        mode="full",
    ):
        self.meta = meta.reset_index(drop=True)
        self.n = len(self.meta)

        self.count = sp.csr_matrix(count)
        self.tfidf = sp.csr_matrix(tfidf)

        self.emb = (
            None if emb is None
            else np.asarray(emb, dtype=np.float32)
        )

        self.cf = (
            None if cf is None
            else np.asarray(cf, dtype=np.float32)
        )

        self.has_cf = (
            np.zeros(self.n, dtype=bool)
            if has_cf is None
            else np.asarray(has_cf, dtype=bool)
        )

        self.mode = mode

        full = self.emb is not None and self.cf is not None

        self.weights = dict(
            weights or (DEFAULT_WEIGHTS if full else LITE_WEIGHTS)
        )

        self.signals = ["count", "tfidf"]

        if self.emb is not None:
            self.signals.append("embed")

        if self.cf is not None:
            self.signals.append("cf")

        self.quality = self.meta["quality"].to_numpy(float)
        self.labels = self.meta["label"].tolist()

        self.label_to_idx = {
            label: i for i, label in enumerate(self.labels)
        }

    # ---------- Save and load ----------

    def save(self, art_dir="artifacts"):
        os.makedirs(art_dir, exist_ok=True)

        meta = self.meta.copy()

        for col in LIST_COLS:
            meta[col] = meta[col].apply(
                lambda values: "|".join(values)
            )

        meta.to_csv(
            os.path.join(art_dir, "meta.csv.gz"),
            index=False,
        )

        sp.save_npz(
            os.path.join(art_dir, "count.npz"),
            self.count,
        )

        sp.save_npz(
            os.path.join(art_dir, "tfidf.npz"),
            self.tfidf,
        )

        if self.emb is not None:
            np.save(
                os.path.join(art_dir, "emb.npy"),
                self.emb,
            )

        if self.cf is not None:
            np.save(
                os.path.join(art_dir, "cf.npy"),
                self.cf,
            )

        np.save(
            os.path.join(art_dir, "has_cf.npy"),
            self.has_cf,
        )

        with open(
            os.path.join(art_dir, "weights.json"),
            "w",
        ) as f:
            json.dump(self.weights, f, indent=2)

    @staticmethod
    def _read_meta(path):
        meta = pd.read_csv(path)

        for col in LIST_COLS:
            meta[col] = (
                meta[col]
                .fillna("")
                .apply(lambda s: [x for x in s.split("|") if x])
            )

        meta["overview"] = meta["overview"].fillna("")
        meta["director"] = meta["director"].fillna("")

        return meta

    @classmethod
    def load(cls, art_dir="artifacts"):
        """Load either Lite artifacts or Full notebook artifacts."""

        meta_path = os.path.join(art_dir, "meta.csv.gz")
        count_path = os.path.join(art_dir, "count.npz")
        tfidf_path = os.path.join(art_dir, "tfidf.npz")

        if not all(
            os.path.exists(p)
            for p in (meta_path, count_path, tfidf_path)
        ):
            raise FileNotFoundError(
                "Required metadata, Count and TF-IDF artifacts are missing."
            )

        weights = None
        weights_path = os.path.join(art_dir, "weights.json")

        if os.path.exists(weights_path):
            with open(weights_path) as f:
                weights = json.load(f)

        emb_path = os.path.join(art_dir, "emb.npy")
        cf_path = os.path.join(art_dir, "cf.npy")
        has_cf_path = os.path.join(art_dir, "has_cf.npy")

        emb = (
            np.load(emb_path, allow_pickle=False)
            if os.path.exists(emb_path)
            else None
        )

        cf = (
            np.load(cf_path, allow_pickle=False)
            if os.path.exists(cf_path)
            else None
        )

        has_cf = (
            np.load(has_cf_path, allow_pickle=False)
            if os.path.exists(has_cf_path)
            else None
        )

        mode = "full" if emb is not None and cf is not None else "lite"

        return cls(
            cls._read_meta(meta_path),
            sp.load_npz(count_path),
            sp.load_npz(tfidf_path),
            emb,
            cf,
            has_cf,
            weights,
            mode=mode,
        )

    @classmethod
    def build_lite(cls, data_dir):
        """Build a Lite model directly from TMDB datasets."""

        import features as ft

        df = ft.load_tmdb(data_dir)
        df, meta, X_count, X_tfidf = ft.build_features(df)

        return cls(
            meta,
            X_count,
            X_tfidf,
            mode="lite",
        )

    # ---------- Similarity calculations ----------

    def components(self, q):
        idxs = _as_list(q)
        out = {}

        for name, matrix in (
            ("count", self.count),
            ("tfidf", self.tfidf),
        ):
            query_vector = np.asarray(
                matrix[idxs].mean(axis=0)
            ).ravel()

            out[name] = np.asarray(
                matrix @ query_vector
            ).ravel()

        if self.emb is not None:
            out["embed"] = (
                self.emb @ self.emb[idxs].mean(axis=0)
            )

        if self.cf is not None:
            valid = [
                i for i in idxs
                if self.has_cf[i]
            ]

            if valid:
                out["cf"] = (
                    self.cf @ self.cf[valid].mean(axis=0)
                )
            else:
                out["cf"] = np.zeros(
                    self.n,
                    dtype=np.float32,
                )

        return out

    # ---------- Filtering ----------

    def filter_mask(
        self,
        genres=None,
        min_rating=0.0,
        language=None,
        year_range=None,
        pool=None,
    ):
        mask = (
            np.ones(self.n, dtype=bool)
            if pool is None
            else pool.copy()
        )

        if genres:
            wanted = set(genres)

            mask &= self.meta["genres_l"].apply(
                lambda g: bool(wanted.intersection(g))
            ).to_numpy()

        if min_rating:
            mask &= (
                self.meta["vote_average"] >= min_rating
            ).to_numpy()

        if language and language != "Any":
            mask &= (
                self.meta["original_language"] == language
            ).to_numpy()

        if year_range:
            mask &= self.meta["year"].between(
                year_range[0],
                year_range[1],
            ).to_numpy()

        return mask

    # ---------- Scoring ----------

    def score(
        self,
        q,
        method="hybrid",
        weights=None,
        mask=None,
        comps=None,
    ):
        idxs = _as_list(q)

        mask = (
            np.ones(self.n, dtype=bool)
            if mask is None
            else mask.copy()
        )

        mask[idxs] = False

        if method == "quality":
            return np.where(mask, self.quality, -np.inf)

        comps = comps or self.components(idxs)

        if method != "hybrid":
            if method not in comps:
                raise ValueError(
                    f"'{method}' is unavailable in {self.mode} mode."
                )

            return np.where(mask, comps[method], -np.inf)

        w = {
            k: v
            for k, v in dict(weights or self.weights).items()
            if k == "quality" or k in comps
        }

        if "cf" in w and not self.has_cf[idxs].any():
            w["cf"] = 0.0

        total = sum(w.values())

        if total <= 0:
            return np.where(mask, self.quality, -np.inf)

        scores = np.zeros(self.n)

        for key in ("tfidf", "embed", "cf"):
            if w.get(key, 0) > 0:
                values = comps[key]

                if key == "cf":
                    values = np.clip(values, 0, None)

                scores += (
                    w[key] / total
                ) * _minmax(values, mask)

        scores += (
            w.get("quality", 0) / total
        ) * self.quality

        return np.where(mask, scores, -np.inf)

    # ---------- Recommendations ----------

    def recommend(
        self,
        movie,
        n=10,
        method="hybrid",
        weights=None,
        **filters,
    ):
        items = (
            movie
            if isinstance(movie, (list, tuple, np.ndarray))
            else [movie]
        )

        idxs = [
            self.label_to_idx[m]
            if isinstance(m, str)
            else int(m)
            for m in items
        ]

        comps = self.components(idxs)
        mask = self.filter_mask(**filters)

        scores = self.score(
            idxs,
            method,
            weights,
            mask,
            comps,
        )

        top = top_n(scores, n)

        result = self.meta.iloc[top].copy()
        result["score"] = scores[top]

        for key, values in comps.items():
            result[f"s_{key}"] = values[top]

        return result

    # ---------- Explanations ----------

    def explain(self, q, j):
        idxs = _as_list(q)
        meta = self.meta

        genres = set()
        keywords = set()
        cast = set()

        directors = {
            meta.at[i, "director"]
            for i in idxs
            if meta.at[i, "director"]
        }

        for i in idxs:
            genres.update(meta.at[i, "genres_l"])
            keywords.update(meta.at[i, "keywords_l"])
            cast.update(meta.at[i, "cast_l"])

        reasons = []

        shared_genres = [
            x for x in meta.at[j, "genres_l"]
            if x in genres
        ]

        if shared_genres:
            reasons.append(
                "Shared genres: " + ", ".join(shared_genres)
            )

        director = meta.at[j, "director"]

        if director and director in directors:
            reasons.append("Same director: " + director)

        shared_cast = [
            x for x in meta.at[j, "cast_l"]
            if x in cast
        ]

        if shared_cast:
            reasons.append(
                "Shared cast: " + ", ".join(shared_cast)
            )

        shared_keywords = [
            x for x in meta.at[j, "keywords_l"]
            if x in keywords
        ][:5]

        if shared_keywords:
            reasons.append(
                "Shared themes: " + ", ".join(shared_keywords)
            )

        return reasons
