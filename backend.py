from flask import Flask, request, jsonify    #flask creates backend server , req receives data from frontend , jsonify sends response in json format
from flask_cors import CORS  #allows frontend to connect to backend 
import numpy as np  #for numerical operations 
import time  #to measure execution time of predictions

from sklearn.datasets import load_iris  #load iris dataset
from sklearn.model_selection import train_test_split #split dataset into training and testing sets

#metrics for evaluation of models or alg
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize   #convert labels to binary format for multi-class ROC curve
from sklearn.linear_model import LogisticRegression  #for classification
from sklearn.neighbors import KNeighborsClassifier   #for classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  #for regression
from sklearn.metrics import mean_squared_error, r2_score

#create flask app instance
app = Flask(__name__) 
#start frontend backend communication
CORS(app)

# CLASSIFICATION SETUP 
iris = load_iris()   #load dataset
X = iris.data   #i/p
y = iris.target  #o/p labels (0,1,2 for 3 classes)
target_names = iris.target_names   #class names (setosa, versicolor, virginica)

#80% TRAINING , 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


lr_model = LogisticRegression(max_iter=200)   #create logistic regression model with max iterations set to 200 to ensure convergence
knn_model = KNeighborsClassifier(n_neighbors=5)   #create KNN model and look at 5 nearest neighbours for classification
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)  #create random forest & use 50 decision trees

#model learns from training data
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)


# HOME - test route to check if backend is running successfully
@app.route("/")
def home():
    return "Backend Running Successfully"



# REGRESSION: LR + RF 
@app.route("/LR_RF", methods=["POST"])   #API endpoint , accepts post requests from frontend and perform LR & RFR
def LR_RF():
    req_data = request.json    #gets json data sent from frontend

    #extract values from json data
    data = req_data.get("data", []) #list of data 
    x_col = req_data.get("xColumn")  #column name for X
    y_col = req_data.get("yColumn")   #column name for Y
    model_type = req_data.get("model", 1)  #model type 1:LR , 2:RFR

    #check if data is received
    if not data:
        return jsonify({"error": "No data received"})

    #check if specified columns exist in data
    if x_col not in data[0] or y_col not in data[0]:
        return jsonify({"error": "Invalid column names"})

     
    #extract x and y values from data
    x_vals = [float(row[x_col]) for row in data]    #convert values to float for numerical processing
    y_vals = [float(row[y_col]) for row in data]    

    #combines x,y and sorts them based on x values to ensure correct order for regression and plotting
    combined = sorted(zip(x_vals, y_vals), key=lambda p: p[0])
    x_vals = [p[0] for p in combined]    #reassign sorted x and y values to x_vals and y_vals
    y_vals = [p[1] for p in combined]    

    #number of data points
    n = len(x_vals)

    # LINEAR REGRESSION
    if model_type == 1:
        sum_x = sum(x_vals)   #sum of x values
        sum_y = sum(y_vals)   #sum of y values
        sum_xy = sum(x_vals[i] * y_vals[i] for i in range(n))   #sum of products of x and y values
        sum_x2 = sum(x * x for x in x_vals)   #sum of squares of x values

        denominator = (n * sum_x2 - sum_x * sum_x)    #denominator for slope calculation in linear regression formula
        if denominator == 0:
            return jsonify({"error": "Division by zero"})

        slope = (n * sum_xy - sum_x * sum_y) / denominator   #calculate slope using linear regression formula
        intercept = (sum_y - slope * sum_x) / n     #calculate intercept using linear regression formula
        predictions = [slope * x + intercept for x in x_vals]   #calculate predicted y values based on slope and intercept

        #line endpoints for plotting regression line
        min_x = min(x_vals)   #minimum x value for plotting regression line
        max_x = max(x_vals)   #maximum x value for plotting regression line
        line_x = [min_x, max_x]   #x values for regression line (from min to max x)
        line_y = [slope * min_x + intercept, slope * max_x + intercept]  #y values for regression line corresponding to min and max x values

        return jsonify({
            "model": 1,
            "x": x_vals,
            "y": y_vals,
            "slope": slope,
            "intercept": intercept,
            "predictions": predictions,
            "line_x": line_x,
            "line_y": line_y
        })

    # RANDOM FOREST REGRESSOR
    elif model_type == 2:
        X_reg = np.array(x_vals).reshape(-1, 1)  #reshape x values to 2D array for regression model
        y_reg = np.array(y_vals)   #convert y values to numpy array for regression model

        trees = req_data.get("trees", 10)    #number of trees in random forest regressor, default is 10 if not specified in request data
        depth = req_data.get("depth", 3)    #maximum depth of each tree in random forest regressor, default is 3 if not specified in request data

        #create and fit random forest regressor model with specified number of trees and depth, and a fixed random state for reproducibility
        reg_model = RandomForestRegressor(
            n_estimators=int(trees),
            max_depth=int(depth),
            random_state=42
        )

        reg_model.fit(X_reg, y_reg)  #fit the random forest regressor model to the data (x and y values)
        predictions = reg_model.predict(X_reg).tolist()   #generate predictions using the fitted model and convert them to a list for JSON serialization

        mse = mean_squared_error(y_reg, predictions)
        r2 = r2_score(y_reg, predictions)

        return jsonify({
            "model": 2,
            "x": x_vals,
            "y": y_vals,
            "predictions": predictions,
            "mse": round(float(mse), 4),
            "r2": round(float(r2), 4),
            "accuracy": round(float(r2 * 100), 2),
            "slope": None,
            "intercept": None,
            "line_x": [],
            "line_y": []
        })

    else:
        return jsonify({"error": "Invalid model type"})


# OVERALL RF ACCURACY 
@app.route("/overall-rf-accuracy", methods=["POST"])     #create API endpoint for calculating overall accuracy of random forest regressor across all pairs of features in the dataset, accepts post requests from frontend
def overall_rf_accuracy():
    req_data = request.json     #get json data from react
    data = req_data["data"]  #extracts actual dataset from json data sent by frontend, expects a key "data" containing the dataset as a list of dictionaries

    columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]   #use these 4 features in combination 
    accuracies = []   #store rsq value of each combination

    #select x
    for x_col in columns:
        #select y
        for y_col in columns:
            #skip where x=y
            if x_col == y_col:
                continue

            x_vals = [float(row[x_col]) for row in data]   #get all values of X & convert to float
            y_vals = [float(row[y_col]) for row in data]   #get all values of Y & convert to float

            X_reg = np.array(x_vals).reshape(-1, 1)  #reshape x values to 2D array (many rows , 1 col)
            y_reg = np.array(y_vals)   #convert y values to array

            #create RFR model
            model = RandomForestRegressor(
                n_estimators=10, max_depth=3, random_state=42
            )


            model.fit(X_reg, y_reg)   #train model
            predictions = model.predict(X_reg)   #generate predictions using the trained model

            r2 = r2_score(y_reg, predictions)   #measure how model fits the data
            accuracies.append(r2 * 100)  #store accuracy as %


    overall_accuracy = sum(accuracies) / len(accuracies)  #avg of comb of all pairs of features
    return jsonify({"overall_accuracy": round(overall_accuracy, 2)})   #convert result to json & round to 2 decimal values
 


# CLASSIFICATION: PREDICT 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json   #get json data from frontend

    #extract feature values and convert to float
    try:
        features = [
            float(data["sepal_length"]),
            float(data["sepal_width"]),
            float(data["petal_length"]),
            float(data["petal_width"])
        ]
    except:
        return jsonify({"error": "Invalid input"}), 400

    input_array = np.array(features).reshape(1, -1)  #convert to numpy array because model expects 2D array (1 row, 4 columns)

    start = time.time()  #start timer to measure prediction time 
    lr_pred = lr_model.predict(input_array)[0]  #predict class using LR model and get the predicted class index (0,1,2)
    lr_time = time.time() - start  #calculate execution time for prediction by taking the difference between current time and start time

    start = time.time()
    knn_pred = knn_model.predict(input_array)[0]
    knn_time = time.time() - start

    start = time.time()
    rf_pred = rf_model.predict(input_array)[0]
    rf_time = time.time() - start

    #send response to frontend
    return jsonify({
        "predictions": {
            "Logistic Regression": target_names[lr_pred],
            "KNN": target_names[knn_pred],
            "Random Forest": target_names[rf_pred]
        },
        "timing": {
            "Logistic Regression": round(lr_time, 6),
            "KNN": round(knn_time, 6),
            "Random Forest": round(rf_time, 6)
        }
    })


# CLASSIFICATION: METRICS 
@app.route("/metrics", methods=["GET"])
def metrics():
    #dictionary storing model name
    models = {
        "Logistic Regression": lr_model,
        "KNN": knn_model,
        "Random Forest": rf_model
    }

    #result dict to store o/p of each model
    result = {}
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])   #ROC curve requires binary format for multi-class classification, so we convert the test labels to a binary format using label_binarize function from sklearn, specifying the classes present in the dataset (0,1,2 for the 3 classes in iris dataset)


    #loop through each model and calculate metrics
    for name, model in models.items():
        y_pred = model.predict(X_test)   #predict labels using test data

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        fpr = {}
        tpr = {}
        roc_auc = {}

        try:
            y_score = model.predict_proba(X_test)
            #loop through 3 classes and calculates ROC for each
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])  #calculate area under curve
        except:
            pass

        #save all metrics for current model in result dict 
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


# RUN 
#app runs only when file executed directly
#debug mode is on for auto reload and better error messages, and server runs on port 5000
if __name__ == "__main__":
    app.run(debug=True, port=5000)