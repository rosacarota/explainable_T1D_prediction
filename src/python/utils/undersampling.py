from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

dataset = pd.read_excel(r'dataset_test.xlsx')

type1_samples = dataset[dataset['Illness'] == 'Type 1 Diabetes']
healthy_samples = dataset[dataset['Illness'] == 'Healthy']

X_type1 = type1_samples.drop('Illness', axis=1)
y_type1 = type1_samples['Illness']

n_healthy_samples = len(healthy_samples)

sss = StratifiedShuffleSplit(n_splits=1,train_size=n_healthy_samples, random_state=42)
for train_index, _ in sss.split(X_type1, y_type1):
    X_type1_resampled = X_type1.iloc[train_index]
    y_type1_resampled = y_type1.iloc[train_index]

X_resampled = pd.concat([X_type1_resampled, healthy_samples.drop('Illness', axis=1)])
y_resampled = pd.concat([y_type1_resampled, healthy_samples['Illness']])

undersampled_dataset = pd.concat([X_resampled, y_resampled], axis=1)

print(undersampled_dataset['Illness'].value_counts())
undersampled_dataset.to_excel(r'undersampled_dataset_test.xlsx', index=False)
print(undersampled_dataset)
