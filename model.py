"""
model.py — Multi-Class Sentiment Analysis System
Classifies text into: Positive | Negative | Neutral | Mixed

Features:
  ✅ Data loading & preprocessing
  ✅ TF-IDF feature engineering
  ✅ Logistic Regression classifier
  ✅ Train/test split (80/20)
  ✅ Full evaluation report (Accuracy, Precision, Recall, F1)
  ✅ Confusion matrix plot
  ✅ Model persistence (joblib)
  ✅ Prediction probabilities
  ✅ Batch prediction
  ✅ Export results to CSV
  ✅ CLI menu system
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

from utils import preprocess

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET     = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model.pkl")
RESULTS_CSV = os.path.join(BASE_DIR, "results.csv")
CM_IMAGE    = os.path.join(BASE_DIR, "confusion_matrix.png")

LABELS = ["Positive", "Negative", "Neutral", "Mixed"]


# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATASET)
    df.columns = df.columns.str.strip()
    df.dropna(inplace=True)
    df["cleaned"] = df["text"].apply(preprocess)
    print(f"✅ Dataset loaded: {len(df)} samples")
    print(df["sentiment"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
# 2. Train model
# ─────────────────────────────────────────────
def train_model(df):
    X = df["cleaned"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"))
    ])

    pipeline.fit(X_train, y_train)
    print("\n✅ Model trained successfully.")

    # ── Evaluation ────────────────────────────
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy: {acc:.2%}")
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS, zero_division=0))

    # ── Confusion matrix ──────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title("Confusion Matrix — Sentiment Classifier", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(CM_IMAGE, dpi=150)
    plt.close()
    print(f"📈 Confusion matrix saved → {CM_IMAGE}")

    # ── Save model ────────────────────────────
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")

    return pipeline


# ─────────────────────────────────────────────
# 3. Load saved model
# ─────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        print("⚠️  No saved model found. Training first …")
        df = load_data()
        return train_model(df)
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
    return pipeline


# ─────────────────────────────────────────────
# 4. Single prediction
# ─────────────────────────────────────────────
def predict_single(pipeline, sentence: str) -> dict:
    cleaned   = preprocess(sentence)
    label     = pipeline.predict([cleaned])[0]
    proba     = pipeline.predict_proba([cleaned])[0]
    classes   = pipeline.classes_
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    return {"text": sentence, "predicted_sentiment": label, "probabilities": prob_dict}


# ─────────────────────────────────────────────
# 5. Batch prediction
# ─────────────────────────────────────────────
def predict_batch(pipeline, sentences: list) -> pd.DataFrame:
    cleaned  = [preprocess(s) for s in sentences]
    labels   = pipeline.predict(cleaned)
    probas   = pipeline.predict_proba(cleaned)
    classes  = pipeline.classes_

    rows = []
    for sent, lbl, prob in zip(sentences, labels, probas):
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(classes, prob)}
        row = {"text": sent, "predicted_sentiment": lbl}
        row.update(prob_dict)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n💾 Batch results exported → {RESULTS_CSV}")
    return df


# ─────────────────────────────────────────────
# 6. CLI Menu
# ─────────────────────────────────────────────
def cli_menu():
    print("\n" + "="*55)
    print("  🤖 Multi-Class Sentiment Analysis System")
    print("="*55)

    pipeline = load_model()

    while True:
        print("\n📌 MENU")
        print("  [1] Predict sentiment of a single sentence")
        print("  [2] Batch predict (enter multiple sentences)")
        print("  [3] Retrain model")
        print("  [4] Exit")
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            sentence = input("\nEnter your sentence: ").strip()
            if not sentence:
                print("❌ Empty input. Try again.")
                continue
            result = predict_single(pipeline, sentence)
            print(f"\n🔍 Text          : {result['text']}")
            print(f"✅ Predicted     : {result['predicted_sentiment']}")
            print("📊 Probabilities :")
            for cls, prob in sorted(result["probabilities"].items(),
                                    key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"   {cls:<10} {prob:.2%}  {bar}")

        elif choice == "2":
            print("\nEnter sentences one per line.")
            print("Type 'DONE' on a new line when finished.\n")
            sentences = []
            while True:
                line = input("> ").strip()
                if line.upper() == "DONE":
                    break
                if line:
                    sentences.append(line)

            if not sentences:
                print("❌ No sentences entered.")
                continue

            df_result = predict_batch(pipeline, sentences)
            print("\n📋 Batch Results:")
            print(df_result[["text", "predicted_sentiment"]].to_string(index=False))

        elif choice == "3":
            df = load_data()
            pipeline = train_model(df)

        elif choice == "4":
            print("\n👋 Goodbye!\n")
            break

        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cli_menu()
