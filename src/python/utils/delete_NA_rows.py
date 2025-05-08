import pandas as pd

df = pd.read_excel(r'C:\Users\rosac\Documents\GitHub\explainable_diabetes_prediction\xlsx\xlsx_GSE9006\platform_1\new\HG-U133A_mapped.xlsx')

df = df.assign(SYMBOL=df['SYMBOL'].fillna('NA'))

df_cleaned = df[df['SYMBOL'] != 'NA']

df_cleaned.to_excel(r'C:\Users\rosac\Documents\GitHub\explainable_diabetes_prediction\xlsx\xlsx_GSE9006\platform_1\new\HG-U133A_mapped_without_NA.xlsx', index=False)
