import pandas as pd
import re

file_path = r"patient_data_without_case_GSE43488.xlsx" 
df = pd.read_excel(file_path)

df = df.map(lambda x: str(x).replace('gender: male', 'M').replace('gender: female', 'F') if isinstance(x, str) else x)

def convert_to_years(value):
    if isinstance(value, str):
        match = re.search(r'age at sample \(months\): (\d+)', value)
        if match:
            months = int(match.group(1))
            years = months // 12
            return years
    return value

df = df.map(convert_to_years)

def diagnosis_replacement(value):
    if isinstance(value, str):
        if "no T1D diagnosis" in value:
            return "Healthy"
        match = re.search(r'(?:time from )?t1d diagnosis \(months\): -?\d+', value)
        if match:
            return "Type 1 Diabetes"
    return value

df = df.map(diagnosis_replacement)

df.to_excel(r"patient_data_cleaned_GSE43488.xlsx", index=False)
