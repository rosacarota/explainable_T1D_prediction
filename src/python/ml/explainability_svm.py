import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import shap
from alibi.explainers import AnchorTabular

df = pd.read_excel(r'undersampled_dataset_threshold.xlsx')

df_cleaned = df.drop(columns=['ID_REF'])

df_cleaned['Illness'] = df_cleaned['Illness'].map({'Healthy': 0, 'Type 1 Diabetes': 1})

X = df_cleaned.drop(columns=['Illness'])
y = df_cleaned['Illness']

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm_model = SVC(random_state=42, probability=True)

param_grid = {
    'C': [100],
    'kernel': ['rbf'],
    'gamma': [3.0]
}

grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=10, scoring='f1', verbose=3)

grid_search.fit(X_train, y_train)

print(f"I migliori parametri trovati sono: {grid_search.best_params_}")

best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

classification_report_output = classification_report(y_test, y_pred, target_names=['Healthy', 'Type 1 Diabetes'])
print("Classification Report:")
print(classification_report_output)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

print("\nMetriche sul Test Set:")
for metric, value in metrics.items():
    print(f"{metric}: {value}")

predict_fn = lambda x: best_svm_model.predict(pd.DataFrame(x, columns=feature_names))

# ** Generazione di spiegazioni tramite Anchor **
explainer = AnchorTabular(predict_fn, feature_names=feature_names)
explainer.fit(X_train.to_numpy(), disc_perc=(25, 50, 75))

for i, sample in enumerate(X_test.to_numpy()):
    explanation = explainer.explain(sample.reshape(1, -1), threshold=0.8)
    prediction = predict_fn(sample.reshape(1, -1))[0]
    
    coverage_percentage = explanation.coverage * 100
    
    print(f"Sample {i}:")
    print(f"Prediction: {prediction}")
    print(f"Anchor: {explanation.anchor}")
    print(f"Precision: {explanation.precision}")
    print(f"Coverage: {coverage_percentage:.4f}%")
    print("\n")

#Inizializza l'explainer di SHAP
explainer = shap.KernelExplainer(best_svm_model.predict_proba, X_train, link="logit")

shap_values = explainer.shap_values(X_test, nsamples=100)

print("SHAP values shape (per classe):", [arr.shape for arr in shap_values])
print("X_test shape:", X_test.shape)

shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names, title="Classe 1 (Type 1 Diabetes)")
