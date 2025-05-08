import pandas as pd
import re

file_path = r"HG-U133A_patients_data_without_T2D.xlsx"
df = pd.read_excel(file_path)

df = df.map(lambda x: str(x).replace('Gender: M', 'M').replace('Gender: F', 'F') if isinstance(x, str) else x)

def extract_age(value):
    if isinstance(value, str):
        match = re.search(r'Age: (\d+) yrs, when sample taken', value)
        if match:
            return int(match.group(1))
    return value

df = df.map(extract_age)

def replace_pbmc_info(value):
    if isinstance(value, str):
        if "PBMCs from Type 2 Diabetes patient" in value:
            return "Type 2 Diabetes"
        elif "PBMCs from Healthy patient" in value:
            return "Healthy"
        elif "PBMCs from Type 1 Diabetes" in value:
            return "Type 1 Diabetes"
    return value

df = df.map(replace_pbmc_info)

df.to_excel(r"HG-U133A_patients_data_cleaned.xlsx", index=False)