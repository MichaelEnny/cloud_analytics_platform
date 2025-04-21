from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from simulate_data import generate_hospital_data

def train_hospital_model():
    df = generate_hospital_data()
    X = df.drop('readmitted', axis=1)
    y = df['readmitted']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    baseline_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=1)
    baseline_model.fit(X_train, y_train)
    baseline_preds = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)

    improved_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2)
    improved_model.fit(X_train, y_train)
    improved_preds = improved_model.predict(X_test)
    improved_accuracy = accuracy_score(y_test, improved_preds)

    print(f"Hospital Baseline Accuracy: {baseline_accuracy:.2f}")
    print(f"Hospital Improved Accuracy: {improved_accuracy:.2f} (+{(improved_accuracy - baseline_accuracy)*100:.1f}%)")

    return baseline_accuracy, improved_accuracy
