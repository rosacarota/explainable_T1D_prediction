import pandas as pd

file1 = r'collapsed_data_HG-U133A.xlsx'
file2 = r'collapsed_data_GSE43488.xlsx'

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

merged_df = pd.merge(df1, df2, on='ID_REF', how='inner')

merged_df.to_excel(r'dataset1_without_metadata.xlsx', index=False)