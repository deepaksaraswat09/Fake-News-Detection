import os
import re
import sys
import joblib

# Paths to trained model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")


def clean_text(text: str) -> str:
    """Preprocess text according to the training pipeline:
    - Lowercase
    - Remove URLs
    - Remove HTML tags
    - Remove non-alphabetic characters
    - Normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(
            f"Model or Vectorizer file not found in {os.path.join(BASE_DIR, 'models')}."
        )
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(news_text: str, model, vectorizer):
    cleaned = clean_text(news_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    # Model probability if supported
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vector)[0]

    # 0 = REAL, 1 = FAKE
    is_fake = bool(prediction == 1)
    label = "FAKE" if is_fake else "REAL"

    return {
        "label": label,
        "is_fake": is_fake,
        "probabilities": {
            "real": probs[0] * 100 if probs is not None else None,
            "fake": probs[1] * 100 if probs is not None else None,
        },
    }


def main():
    print("=" * 60)
    print("      📰 FAKE NEWS DETECTION SYSTEM (ML Inference)      ")
    print("=" * 60)

    try:
        model, vectorizer = load_artifacts()
        print("✓ Model and TF-IDF Vectorizer loaded successfully.\n")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        sys.exit(1)

    if len(sys.argv) > 1:
        # CLI argument mode
        news_input = " ".join(sys.argv[1:])
        result = predict(news_input, model, vectorizer)
        print(f"Input: {news_input[:100]}...")
        print(f"Prediction: [{result['label']}]")
        if result["probabilities"]["real"] is not None:
            print(
                f"Confidence -> Real: {result['probabilities']['real']:.2f}% | Fake: {result['probabilities']['fake']:.2f}%"
            )
        return

    # Interactive loop mode
    print("Type a news headline or paragraph (or 'exit' / 'quit' to stop):\n")
    while True:
        try:
            user_input = input("Enter news text > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting. Thank you!")
                break

            res = predict(user_input, model, vectorizer)
            badge = "🚨 FAKE NEWS" if res["is_fake"] else "✅ REAL NEWS"
            print("-" * 50)
            print(f"Result: {badge}")
            if res["probabilities"]["real"] is not None:
                print(
                    f"Confidence: Real = {res['probabilities']['real']:.2f}% | Fake = {res['probabilities']['fake']:.2f}%"
                )
            print("-" * 50 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you!")
            break


if __name__ == "__main__":
    main()
