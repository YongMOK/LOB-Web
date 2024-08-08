import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#from Neural_network import SLFN
import joblib

def feature_for_k_events(data, k):
    data = data.iloc[:, :144]
    num_new_rows = data.shape[0] - k + 1
    num_new_cols = data.shape[1] * k
    df_new = np.zeros((num_new_rows, num_new_cols))

    for i in range(num_new_rows):
        df_new[i] = data.iloc[i:i+k, :].values.flatten()
    return pd.DataFrame(df_new)

def label_k_event(mid_price, k):
    length = mid_price.shape[0]
    label = np.zeros(length-k+1)
    for i in range(length-k):
        percentage_change = (mid_price[i + k] -  mid_price[i])/abs(mid_price[i])
        if percentage_change >= 0.000002:
            label[i] = 1
        elif -0.000002 <= percentage_change < 0.000002:
            label[i] = 2
        elif percentage_change < -0.000002:
            label[i] = 3
    label[length-k] = 3
    return pd.DataFrame(label)

def get_mid_price(data):
    return data.iloc[:, 41]

def train_test_with_uploaded_file(model_name, training_data, testing_data, k):
    try:
        training_mid_price = get_mid_price(training_data)
        testing_mid_price = get_mid_price(testing_data)
        
        X_train = feature_for_k_events(training_data, k)
        Y_train = label_k_event(training_mid_price, k)
        
        X_test = feature_for_k_events(testing_data, k)
        Y_test = label_k_event(testing_mid_price, k)
        
        # Select the model
        model_dict = {
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Ridge Regression": RidgeClassifier(alpha=1.0),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=50),
        }
        
        model = model_dict.get(model_name)
        if model is None:
            return {"error": "Model not found"}
        
        # Train the model
        model.fit(X_train, Y_train.values.ravel())
        Y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(Y_test, Y_pred)
        class_report = classification_report(Y_test, Y_pred, output_dict=True)
        conf_matrix = confusion_matrix(Y_test, Y_pred).tolist()
        
        # Save the model for future use
        model_filename = f"{model_name}_model.joblib"
        joblib.dump(model, model_filename)
        
        return {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix,
            "model_filename": model_filename
        }
    except Exception as e:
        
        return {"error": str(e)}

def 