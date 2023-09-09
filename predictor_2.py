import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, \
                            average_precision_score, precision_recall_curve, confusion_matrix, \
                            matthews_corrcoef, cohen_kappa_score

path1 = "C:/Users/alvar/PycharmProjects/Proyecto2/data_npy"

print(f"Loading data...")
X_train = np.load(f"{path1}/X_train_mfcc.npy")
y_train = np.load(f"{path1}/y_train.npy")
X_test = np.load(f"{path1}/X_test_mfcc.npy")
y_test = np.load(f"{path1}/y_test.npy")
print(f"Data loaded")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Fitting LogisticRegression model...")
clf1 = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
print(f"Model LogisticRegression fitted")

# Predecir las etiquetas en los datos de prueba
print(f"Predicting LogisticRegression...")
y_pred1 = clf1.predict(X_test)
print(f"Predicted LogisticRegression.")


LogisticRegression_measures = {
    'Model': ['Logistic Regression'],
    'accuracy': accuracy_score(y_test, y_pred1),
    'f1_score': f1_score(y_test, y_pred1),
    'precision': precision_score(y_test, y_pred1),
    'recall': recall_score(y_test, y_pred1),
    'roc_auc': roc_auc_score(y_test, y_pred1),
    'average_precision': average_precision_score(y_test, y_pred1),
    'confusion_matrix': confusion_matrix(y_test, y_pred1).tolist(),
    'matthews_corrcoef': matthews_corrcoef(y_test, y_pred1),
    'cohen_kappa_score': cohen_kappa_score(y_test, y_pred1)
}

pathDiccionarios = "C:/Users/alvar/PycharmProjects/Proyecto2/diccionarios"
with open(f"{pathDiccionarios}/LogisticRegression.json", 'w') as json_file1:
    json.dump(LogisticRegression_measures, json_file1, indent=4)


print(f"Creating RandomForestRegressor model...")
regr = RandomForestRegressor(random_state=0, n_estimators=10)
print(f"RandomForestRegressor created")


print(f"Fitting RandomForestRegressor model...")
regr.fit(X_train, y_train)
#clf2 = RandomForestRegressor(random_state=0).fit(X_train, y_train)
print(f"Model RandomForestRegressor fitted")

# Predecir las etiquetas en los datos de prueba usando RandomForestRegressor
print(f"Predicting RandomForestRegressor...")
y_pred2 = regr.predict(X_test)
print(f"Predicted RandomForestRegressor.")

f1_RandomForestRegressor = f1_score(y_test, y_pred2)
print(f"F1-score RandomForestRegressor: {f1_RandomForestRegressor:.2f}")



f1_LogisticRegression= f1_score(y_test, y_pred1)
print(f"F1-score LogisticRegression: {f1_LogisticRegression:.2f}")

RandomForestRegressor_measures = {
    'Model': 'RandomForestRegressor',
    'accuracy': accuracy_score(y_test, y_pred2),
    'f1_score': f1_score(y_test, y_pred2),
    'precision': precision_score(y_test, y_pred2),
    'recall': recall_score(y_test, y_pred2),
    'roc_auc': roc_auc_score(y_test, y_pred2),
    'average_precision': average_precision_score(y_test, y_pred2),
    'confusion_matrix': confusion_matrix(y_test, y_pred2).tolist(),
    'matthews_corrcoef': matthews_corrcoef(y_test, y_pred2),
    'cohen_kappa_score': cohen_kappa_score(y_test, y_pred2)
}


# Guardar diccionarios
with open(f"{pathDiccionarios}/RandomForestRegressor.json", 'w') as json_file2:
    json.dump(RandomForestRegressor_measures, json_file2, indent=4)
