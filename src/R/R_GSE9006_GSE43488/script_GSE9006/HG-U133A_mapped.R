if(!require(readxl)) install.packages("readxl")
if(!require(hgu133b.db)) BiocManager::install("hgu133b.db")
if(!require(dplyr)) install.packages("dplyr")
if(!require(openxlsx)) install.packages("openxlsx")

library(readxl)
library(hgu133a.db)
library(dplyr)
library(openxlsx)
library(AnnotationDbi)

dataset_path <- "HG-U133A_normalized_without_T2D.xlsx"
df <- read_excel(dataset_path)

probe_to_gene <- AnnotationDbi::select(hgu133a.db, keys = df$Probe_ID, columns = "SYMBOL", keytype = "PROBEID")

merged_df <- merge(df, probe_to_gene, by.x = "Probe_ID", by.y = "PROBEID", all.x = TRUE)

merged_data_filtered <- merged_df[grepl("(^[0-9]+(_at|_a_at|_s_at)$)", merged_df$Probe_ID), ]

merged_data_filtered <- merged_data_filtered %>%
  group_by(Probe_ID) %>%
  mutate(Probe_ID = paste0(Probe_ID, "_", row_number())) %>%
  ungroup()

write.xlsx(merged_data_filtered, "HG-U133A_mapped.xlsx") 
