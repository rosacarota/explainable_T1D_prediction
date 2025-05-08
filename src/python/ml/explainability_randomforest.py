import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from alibi.explainers import AnchorTabular
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel(r'undersampled_dataset_threshold.xlsx')

df_cleaned = df.drop(columns=['ID_REF'])
X = df_cleaned.drop(columns=['Illness'])
y = df_cleaned['Illness']

y = y.map({'Healthy': 0, 'Type 1 Diabetes': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

feature_names = X.columns.tolist()

rf_model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [400],
    'max_depth': [25],
    'min_samples_split': [6],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

print("Migliori iperparametri trovati:")
print(grid_search.best_params_)

best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

classification_report_output = classification_report(y_test, y_pred, target_names=['Healthy', 'Type 1 Diabetes'])
print("\nClassification Report:")
print(classification_report_output)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

print("\nMetriche Finali:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

# ** Generazione di spiegazioni tramite Anchor **

predict_fn = lambda x: best_rf_model.predict(pd.DataFrame(x, columns=X.columns))

feature_names = X.columns.tolist()
explainer = AnchorTabular(predict_fn, feature_names=feature_names, categorical_names={})
explainer.fit(X_train.values, disc_perc=(25, 50, 75))

for idx in range(len(X_test)):
    instance = X_test.iloc[idx].values.reshape(1, -1)
    explanation = explainer.explain(instance)

    prediction = best_rf_model.predict(instance)[0]

    print(f"\nIstanza {idx + 1}:")
    print(f"Predizione del modello: {prediction}")
    print('Regola (Anchor):', ' AND '.join(explanation.anchor))
    print(f"Precisione dell'Anchor: {explanation.precision:.2f}")
    print(f"Copertura dell'Anchor: {explanation.coverage:.2f}")

y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]

# Calcolo dell'AUC-ROC
auc = roc_auc_score(y_test_encoded, y_pred_proba)
print("\nAUC-ROC:", auc)

fpr, tpr, thresholds = roc_curve(y_test_encoded, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label='AUC-ROC (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'k--')  # Linea diagonale
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print(feature_names)

#Inizializza l'explainer di SHAP per i modelli basati su alberi
explainer = shap.TreeExplainer(best_rf_model)

shap_values = explainer.shap_values(X_test)

print("Shape of X_test:", X_test.shape)
print("Shape of shap_values:", shap_values.shape)


shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names)