import pandas as pd

metadata = pd.read_excel(r"dataset1_with_T2D.xlsx")

metadata_t = metadata.T

if 'ID_REF' in metadata_t.index:
    metadata_t.columns = metadata_t.loc['ID_REF']
    metadata_t = metadata_t.drop('ID_REF') # Rimozione di ID_REF dalle righe 

# Rimuove le righe con 'Type 2 Diabetes' come etichetta in Illness
if 'Illness' in metadata_t.columns:
    metadata_t_filtered = metadata_t[metadata_t['Illness'] != 'Type 2 Diabetes']
else:
    metadata_t_filtered = metadata_t

metadata_t_filtered['ID_REF'] = metadata_t_filtered.index

cols = ['ID_REF'] + [col for col in metadata_t_filtered.columns if col != 'ID_REF']
metadata_t_filtered = metadata_t_filtered[cols]

metadata_t_filtered.to_excel(r"dataset1.xlsx", index=False)