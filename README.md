# 🤖 Multi-Class Sentiment Analysis System

**AI Intern — Task 1 | Hasnain Karimain Educational Academy**

Classifies text into **four sentiment categories**:
`Positive` | `Negative` | `Neutral` | `Mixed`

---

## 📁 Folder Structure

```
Sentiment_Classifier/
├── dataset.csv          # 100 labeled sentences (4 classes)
├── model.py             # Main script: train, evaluate, predict, CLI
├── utils.py             # Text preprocessing utilities
├── model.pkl            # Saved model (generated after training)
├── results.csv          # Batch prediction output (generated on use)
├── confusion_matrix.png # Evaluation plot (generated after training)
└── README.md
```

---

## 🛠 Requirements

```bash
pip install pandas scikit-learn nltk matplotlib seaborn joblib
```

---

## 🚀 How to Run

```bash
python model.py
```

You will see an interactive menu:

```
📌 MENU
  [1] Predict sentiment of a single sentence
  [2] Batch predict (enter multiple sentences)
  [3] Retrain model
  [4] Exit
```

---

## ✅ Features Implemented

| Feature | Status |
|---|---|
| 4-class classification (Pos/Neg/Neu/Mixed) | ✅ |
| Text preprocessing (lowercase, punctuation, stopwords, lemmatization) | ✅ |
| TF-IDF feature engineering (unigrams + bigrams) | ✅ |
| Logistic Regression classifier | ✅ |
| 80/20 train/test split | ✅ |
| Accuracy, Precision, Recall, F1 evaluation | ✅ |
| Confusion Matrix (saved as PNG) | ✅ (Bonus) |
| Model saved with joblib | ✅ (Bonus) |
| Prediction probabilities shown | ✅ (Bonus) |
| Batch prediction | ✅ (Bonus) |
| CLI menu system | ✅ (Bonus) |
| Export results to CSV | ✅ (Bonus) |

---

## 📊 Example

**Input:**
```
Enter your sentence: I love the design but hate the performance
```

**Output:**
```
🔍 Text          : I love the design but hate the performance
✅ Predicted     : Mixed
📊 Probabilities :
   Mixed      72.31%  ██████████████████████
   Negative   14.10%  ████
   Positive   10.44%  ███
   Neutral     3.15%  █
```

---

## 🧠 Tech Stack

- **Language:** Python 3
- **ML:** scikit-learn (Logistic Regression, TF-IDF, Pipeline)
- **NLP:** NLTK (stopwords, lemmatization, tokenization)
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** joblib
