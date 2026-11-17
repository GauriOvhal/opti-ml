from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# ================= LOAD DATA =================
iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= TRAIN MODELS =================
lr_model = LogisticRegression(max_iter=200)
knn_model = KNeighborsClassifier(n_neighbors=5)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# ================= HOME =================
@app.route("/")
def home():
    return "Iris Classification Backend Running ✅"

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    try:
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]
    except:
        return jsonify({"error": "Invalid input"}), 400

    input_array = np.array(features).reshape(1, -1)

    results = {}

    # Logistic Regression
    start = time.time()
    lr_pred = lr_model.predict(input_array)[0]
    lr_time = time.time() - start

    # KNN
    start = time.time()
    knn_pred = knn_model.predict(input_array)[0]
    knn_time = time.time() - start

    # Random Forest
    start = time.time()
    rf_pred = rf_model.predict(input_array)[0]
    rf_time = time.time() - start

    results["predictions"] = {
        "Logistic Regression": target_names[lr_pred],
        "KNN": target_names[knn_pred],
        "Random Forest": target_names[rf_pred]
    }

    results["timing"] = {
        "Logistic Regression": round(lr_time, 6),
        "KNN": round(knn_time, 6),
        "Random Forest": round(rf_time, 6)
    }

    return jsonify(results)

# ================= METRICS =================
@app.route("/metrics", methods=["GET"])
def metrics():
    models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Random Forest": rf_model
    }

    result = {}

    # For ROC (multi-class)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Basic metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        # ROC Curve
        fpr = {}
        tpr = {}
        roc_auc = {}

        try:
            y_score = model.predict_proba(X_test)

            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        except:
            pass

        result[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist(),
            "roc": {
                "fpr": {str(k): v.tolist() for k, v in fpr.items()},
                "tpr": {str(k): v.tolist() for k, v in tpr.items()},
                "auc": roc_auc
            }
        }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)