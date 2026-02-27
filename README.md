🎬 IMDB Sentiment Analysis — Machine Learning NLP Pipeline
A complete, production‑ready sentiment analysis system built on the IMDB movie reviews dataset.
This project implements a full end‑to‑end workflow including data preprocessing, text normalization, model training with hyperparameter tuning, evaluation with visual reports, and real‑time prediction on new text inputs.

⭐ Project Highlights
End‑to‑end NLP pipeline using classical ML models

TF‑IDF + ML classifiers inside a unified Pipeline

GridSearchCV for hyperparameter optimization

Text preprocessing with tokenization + lemmatization

Evaluation reports with Confusion Matrix, ROC Curve, and Classification Report

Model persistence using Joblib

Interactive prediction script (predict.py)

Clean, modular, industry‑standard project structure

📂 Project Structure

sentiment_analysis_project/
│

├── data/

│   └── raw/

│       └── IMDB Dataset.csv

│
├── models/

│   └── best_model.pkl

│

├── reports/

│   └── plots/

│       ├── confusion_matrix.png

│       ├── confusion_matrix_normalized.png

│       └── roc_curve.png

│

├── src/

│   ├── data_preprocessing.py

│   ├── text_preprocessing.py

│   ├── train_models.py

│   ├── evaluation.py

│   └── predict.py

│
├── main.py

└── README.md



🧹 Text Preprocessing
The preprocessing pipeline includes:

Lowercasing

Removing HTML tags

Removing URLs

Removing punctuation

Normalizing whitespace

Tokenization (word_tokenize)

Lemmatization (WordNetLemmatizer)

This ensures clean, normalized text for TF‑IDF vectorization.

🤖 Models Trained
Three machine learning models were trained and tuned:

Logistic Regression

Linear SVM (LinearSVC)

Multinomial Naive Bayes

Each model is wrapped inside a Pipeline:
TF-IDF Vectorizer → Classifier

Hyperparameters are optimized using GridSearchCV with 3‑fold cross‑validation.

🏆 Model Performance
Model	              Accuracy	    F1‑Score
Logistic Regression	0.9007	      0.9016
Linear SVM	        0.8991	      0.9005
Naive Bayes	        0.8698	      0.8716
✔ Best Model: Logistic Regression

📊 Evaluation Results
Classification Report

              precision    recall  f1-score   support

    Negative       0.91      0.89      0.90      5000
    Positive       0.89      0.91      0.90      5000

    accuracy                           0.90     10000
   macro avg       0.90      0.90      0.90     10000
weighted avg       0.90      0.90      0.90     10000


Saved Plots (in reports/plots/)
confusion_matrix.png

confusion_matrix_normalized.png

roc_curve.png

These are automatically generated during evaluation.

💾 Saving the Best Model
The best model is saved automatically:
models/best_model.pkl

This allows fast loading for inference without retraining.

🔮 Predicting Sentiment on New Text
Use the interactive prediction script:
python src/predict.py

Example:
Enter a review: this movie was absolutely amazing!
Sentiment: Positive
Confidence: 0.8732

🛠 How to Run the Project

1) Install dependencies
pip install -r requirements.txt

2) Train models and generate evaluation reports
python main.py

3) Predict sentiment for new text
python src/predict.py

📦 Requirements
pandas
numpy
matplotlib
sklearn
seaborn
os
joblib
re
string
nltk

