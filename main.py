"""
To-Do Task Categorizer
Predicts task priority and estimated time (in minutes) from task descriptions.
Users can correct predictions; corrections are saved to data.csv and used in later training.
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = Path(__file__).with_name("data.csv")
RANDOM_STATE = 42

def duration_category(minutes):
    if minutes < 30:
        return "short"
    if minutes <= 90:
        return "medium"
    return "long"

def make_classifier():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

def make_regressor():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)),
        ("regressor", Ridge(alpha=1.0))
    ])

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    data = pd.read_csv(DATA_FILE)
    required = {"description", "priority", "expected_minutes"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    data = data.dropna(subset=list(required)).copy()
    data["description"] = data["description"].astype(str).str.strip()
    data["priority"] = data["priority"].astype(str).str.strip().str.lower()
    data["expected_minutes"] = pd.to_numeric(data["expected_minutes"], errors="coerce")
    data = data.dropna(subset=["expected_minutes"])
    data = data[data["description"].ne("")]
    data = data[data["priority"].isin(["low", "medium", "high"])]
    data = data[data["expected_minutes"] > 0]
    if len(data) < 6:
        raise ValueError("Not enough valid rows to train. Add at least 6 examples.")
    data["duration_category"] = data["expected_minutes"].apply(duration_category)
    return data

def evaluate_model(X, y, name):
    if y.nunique() < 2 or y.value_counts().min() < 2:
        print(f"\n{name}: evaluation skipped (not enough examples per class).")
        return
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    model = make_classifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name} test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print(classification_report(y_test, predictions, zero_division=0))

def train_models(data):
    X = data["description"]
    priority_model = make_classifier().fit(X, data["priority"])
    duration_model = make_classifier().fit(X, data["duration_category"])
    time_model = make_regressor().fit(X, data["expected_minutes"])
    return priority_model, duration_model, time_model

def save_feedback(description, priority, minutes):
    """Append the user's corrected labels to the CSV training data."""
    feedback_row = pd.DataFrame([{
        "description": description,
        "priority": priority,
        "expected_minutes": minutes
    }])
    feedback_row.to_csv(DATA_FILE, mode="a", header=False, index=False, encoding="utf-8")

def main():
    data = load_data()
    print(f"Loaded {len(data)} usable task examples.")

    evaluate_model(data["description"], data["priority"], "Priority")
    evaluate_model(data["description"], data["duration_category"], "Duration")

    priority_model, duration_model, time_model = train_models(data)
    print("\nTask categorizer ready. Type a task; enter 'quit' to exit.")

    while True:
        task = input("\nTask description: ").strip()
        if task.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not task:
            print("Please enter a non-empty task description.")
            continue

        priority = priority_model.predict([task])[0]
        category = duration_model.predict([task])[0]
        estimated_minutes = max(1, int(round(float(time_model.predict([task])[0]))))

        print(f"Predicted priority: {priority}")
        print(f"Estimated time: about {estimated_minutes} minutes ({category})")

        answer = input("Are these predictions correct? (y/n, Enter to skip feedback): ").strip().lower()
        if answer in {"y", "yes", "n", "no"}:
            corrected_priority = priority
            corrected_minutes = estimated_minutes
            if answer in {"n", "no"}:
                corrected_priority = input("Correct priority (low/medium/high): ").strip().lower()
                while corrected_priority not in {"low", "medium", "high"}:
                    corrected_priority = input("Please enter low, medium, or high: ").strip().lower()
                while True:
                    try:
                        corrected_minutes = int(input("Correct estimated time in minutes: ").strip())
                        if corrected_minutes > 0:
                            break
                        print("Minutes must be greater than zero.")
                    except ValueError:
                        print("Please enter a whole number.")
            save_feedback(task, corrected_priority, corrected_minutes)
            data = load_data()
            priority_model, duration_model, time_model = train_models(data)
            print("Feedback saved to data.csv. Models updated for future tasks.")

if __name__ == "__main__":
    main()
