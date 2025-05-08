import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular

file_path = r'undersampled_dataset_threshold.xlsx'
dataset = pd.read_excel(file_path)

X = dataset.drop(columns=['Illness', 'ID_REF'])
y = dataset['Illness']

y = y.map({'Healthy': 0, 'Type 1 Diabetes': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

param_grid = {
    'colsample_bytree': [0.5],
    'learning_rate': [0.01],
    'max_depth': [15],
    'n_estimators': [150],
    'subsample': [0.6],
    'min_child_weight': [1],
    'gamma': [0.1],
    'reg_alpha': [0],
    'reg_lambda': [1]
}

model = XGBClassifier(tree_method='hist', random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=10, verbose=3)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f'Best Parameters: {grid_search.best_params_}')

y_pred = best_model.predict(X_test)

classification_report_output = classification_report(y_test, y_pred, target_names=['Healthy', 'Type 1 Diabetes'])
print("Classification Report:")
print(classification_report_output)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')

y_pred_proba = best_model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred_proba)
print("\nAUC-ROC:", auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC-ROC (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Linea diagonale
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# ** Generazione di spiegazioni tramite Anchor **
feature_names = X.columns.tolist()
predict_fn = lambda x: best_model.predict(pd.DataFrame(x, columns=feature_names))

explainer = AnchorTabular(predict_fn, feature_names=feature_names)
explainer.fit(X_train.to_numpy(), disc_perc=(25, 50, 75))

for i, sample in enumerate(X_test.to_numpy()):
    explanation = explainer.explain(sample.reshape(1, -1))
    
    prediction = predict_fn(sample.reshape(1, -1))[0]
    
    coverage_percentage = explanation.coverage * 100
    
    print(f"Sample {i}:")
    print(f"Prediction: {prediction}")
    print(f"Anchor: {explanation.anchor}")
    print(f"Precision: {explanation.precision}")
    print(f"Coverage: {coverage_percentage:.4f}%")
    print(f"True Label: {y_test.iloc[i]}")
    print("\n")

#Inizializza l'explainer di SHAP per i modelli basati su alberi
explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

shap.summary_plot(shap_values, X_test, feature_names=feature_names)