"""
Step 2d: The main lifecycle classifier (FR04).

1. Turn every activity's text into a 384-number "meaning vector" with the
   all-MiniLM-L6-v2 sentence-transformer (texts with similar meaning get
   similar vectors, even with different words).
2. Train a Logistic Regression on the gold-labelled vectors.
3. Test it fairly with 5-fold cross-validation and compare with the keyword
   baseline on exactly the same items (macro F1, target >= 0.75).
4. Label ALL activities with stage + confidence and save as labeller 'model'.

Run from backend:
    python -m component2.classify.train_model
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import joblib

from component2.config import MODEL_DIR
from component2.db import connect, save_labels

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_CHARS = 1000            # MiniLM reads ~256 words; longer text is cut anyway


def model_input(source, text):
    return f"[{source}] {text[:MAX_CHARS]}"     # telling the model where text came from helps it


def get_embeddings(conn):
    """Embed every activity once and cache the result, so re-runs take seconds."""
    rows = conn.execute("SELECT activity_id, source, text FROM activities ORDER BY activity_id").fetchall()
    ids = [r[0] for r in rows]
    cache = MODEL_DIR / "embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if list(data["ids"]) == ids:
            print(f"Using cached embeddings for {len(ids)} activities")
            return ids, data["vectors"]

    from sentence_transformers import SentenceTransformer
    print(f"Embedding {len(ids)} activities with {MODEL_NAME} (first time takes a few minutes)...")
    encoder = SentenceTransformer(MODEL_NAME)
    vectors = encoder.encode([model_input(s, t) for _, s, t in rows],
                             batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids, dtype=object), vectors=vectors)
    return ids, vectors


def main():
    conn = connect()
    ids, vectors = get_embeddings(conn)
    row_of = {aid: n for n, aid in enumerate(ids)}

    gold = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='gold'").fetchall())
    baseline = dict(conn.execute("SELECT activity_id, stage FROM stage_labels WHERE labeller='baseline'").fetchall())
    gold_ids = [i for i in gold if i in row_of]
    if len(gold_ids) < 50:
        print(f"Only {len(gold_ids)} gold labels - run load_ratings first.")
        return

    X = vectors[[row_of[i] for i in gold_ids]]
    y = np.array([gold[i] for i in gold_ids])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    # Fair test: each item is predicted by a model that never saw it
    n_splits = max(2, min(5, int(np.unique(y, return_counts=True)[1].min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    y_base = np.array([baseline.get(i, "other") for i in gold_ids])

    f1_model = f1_score(y, y_pred, average="macro", zero_division=0)
    f1_base = f1_score(y, y_base, average="macro", zero_division=0)
    print(f"\nGold items: {len(gold_ids)}   ({n_splits}-fold cross-validation)")
    print(f"Macro F1  keyword baseline : {f1_base:.3f}")
    print(f"Macro F1  MiniLM + LogReg  : {f1_model:.3f}   (target >= 0.75)")
    print("\nPer-stage results for the model:")
    print(classification_report(y, y_pred, zero_division=0))

    # Final model: train on all gold labels, then label every activity
    clf.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "stage_classifier.joblib")
    probs = clf.predict_proba(vectors)
    stages = clf.classes_[probs.argmax(axis=1)]
    conf = probs.max(axis=1)
    save_labels(conn, "model", [(ids[n], stages[n], round(float(conf[n]), 3)) for n in range(len(ids))])
    low = int((conf < 0.5).sum())
    print(f"Labelled all {len(ids)} activities. Low-confidence (< 0.5): {low} -> candidates for LLM check")


if __name__ == "__main__":
    main()