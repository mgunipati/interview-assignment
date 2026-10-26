"""Unsupervised theme discovery.

Two uses:
- baseline: cluster whole calls (TF-IDF and embeddings) to show what clustering alone finds;
- discovery: UMAP + HDBSCAN over LLM-extracted issue statements (the BERTopic recipe,
  without the extra dependency), with class-based TF-IDF terms to describe each cluster.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import normalize

SEED = 42
# Conversational filler that dominates raw transcripts but carries no topic
FILLER = {"yeah", "okay", "ok", "um", "uh", "like", "just", "right", "know", "think", "really", "gonna",
          "mean", "kind", "sort", "actually", "got", "lot", "thing", "things", "going", "want", "good",
          "great", "sure", "let", "make", "need", "able", "way", "today", "customer", "aegis", "id"}


def stop_words(extra=()):
    return list(ENGLISH_STOP_WORDS | FILLER | {w.lower() for w in extra})


def tfidf_lsa(docs, extra_stop=(), n_components=50):
    vec = TfidfVectorizer(stop_words=stop_words(extra_stop), ngram_range=(1, 2), min_df=2, max_df=0.6,
                          sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = vec.fit_transform(docs)
    Z = TruncatedSVD(n_components=min(n_components, X.shape[1] - 1), random_state=SEED).fit_transform(X)
    return normalize(Z), vec, X


def sweep_kmeans(Z, ks=range(4, 15)):
    """Silhouette for each k: an honest read of how separable the data is."""
    rows = []
    for k in ks:
        labels = KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, labels, metric="cosine")})
    return pd.DataFrame(rows)


def kmeans(Z, k):
    return KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)


def top_terms(X, vec, labels, n=8):
    """Highest mean TF-IDF terms per cluster."""
    terms = np.array(vec.get_feature_names_out())
    return {c: list(terms[np.asarray(X[labels == c].mean(axis=0)).ravel().argsort()[::-1][:n]])
            for c in sorted(set(labels))}


def ctfidf_terms(texts, labels, n=8, extra_stop=()):
    """Class-based TF-IDF (BERTopic): treat each cluster as one document."""
    df = pd.DataFrame({"t": texts, "c": labels})
    df = df[df.c >= 0].groupby("c").t.apply(" ".join)
    cv = CountVectorizer(stop_words=stop_words(extra_stop), ngram_range=(1, 2), token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    tf = cv.fit_transform(df.values).toarray().astype(float)
    tf = tf / tf.sum(axis=1, keepdims=True)
    idf = np.log(1 + tf.shape[0] / (tf > 0).sum(axis=0))
    w, terms = tf * idf, cv.get_feature_names_out()
    return {c: list(terms[w[i].argsort()[::-1][:n]]) for i, c in enumerate(df.index)}


def umap_hdbscan(E, n_neighbors=10, n_components=5, min_cluster_size=6, min_samples=3):
    import umap
    U = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric="cosine",
                  min_dist=0.0, random_state=SEED).fit_transform(E)
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(U)
    return labels, U


def umap_2d(E, n_neighbors=10):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, n_components=2, metric="cosine", min_dist=0.1,
                     random_state=SEED).fit_transform(E)


def alignment(labels, reference):
    """How much a clustering just reproduces a known label (ARI + purity)."""
    ct = pd.crosstab(labels, reference)
    return {"ARI": adjusted_rand_score(reference, labels), "purity": ct.max(axis=1).sum() / ct.values.sum()}
