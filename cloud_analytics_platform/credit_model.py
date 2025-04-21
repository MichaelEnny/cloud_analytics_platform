from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from simulate_data import generate_credit_data

def train_credit_model():
    df = generate_credit_data()
    X = df.drop('defaulted', axis=1)
    y = df['defaulted']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    baseline_model = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    improved_model = XGBClassifier(n_estimators=100, max_depth=6, verbosity=0)
    improved_model.fit(X_train, y_train)
    improved_preds = improved_model.predict(X_test)
    improved_accuracy = accuracy_score(y_test, improved_preds)

    print(f"Credit Baseline Accuracy: {baseline_accuracy:.2f}")
    print(f"Credit Improved Accuracy: {improved_accuracy:.2f} (+{(improved_accuracy - baseline_accuracy)*100:.1f}%)")

    return baseline_accuracy, improved_accuracy
