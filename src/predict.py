import joblib
import os
from src.text_preprocessing import TextPreprocessor

def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models" , "best_model.pkl")
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    return model

def predict_sentence(model, text:str , tp: TextPreprocessor):
    # Preprocess text
    clean_text = tp.clean_text(text)
    clean_text = tp.lemmatize_text(clean_text)

    # Predict
    prediction = model.predict([clean_text])[0]

    # Probability (if available)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba([clean_text])[0][1]

    else:
        # fallback for SVM
        prob = model.decision_function([clean_text])[0]

    label = "Positive" if prediction == 1 else "Negative"

    return label , prob

if __name__ == "__main__":
    # Load model
    model = load_model()

    # Create preprocessor once (performance boost)
    tp = TextPreprocessor()

    while True:
        text = input("\nEnter a review (or type 'exit'): ")

        if text.lower() == "exit":
            break
        label , confidence = predict_sentence(model, text , tp)

        print(f"Sentiment: {label}")
        print(f"Confidence: {confidence:.4f}")