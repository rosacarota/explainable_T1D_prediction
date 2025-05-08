library(writexl)
library(hgu219.db)

matrix_data <- read.table("explainable_diabetes_prediction\\txt\\GSE43488_series_matrix.txt",
                          sep = "\t", 
                          header = TRUE, 
                          fill = TRUE, 
                          comment.char = "!")


gene_mapping <- select(
  hgu219.db,
  keys = matrix_data$ID_REF, 
  columns = c("SYMBOL", "GENENAME"),
  keytype = "PROBEID"
)

gene_mapping_combined <- aggregate(
  cbind(SYMBOL, GENENAME) ~ PROBEID, 
  data = gene_mapping, 
  FUN = function(x) paste(unique(x), collapse = ", ")
)

merged_data <- merge(matrix_data, gene_mapping, by.x = "ID_REF", by.y = "PROBEID", all.x = TRUE)

merged_data_filtered <- merged_data[grepl("(^[0-9]+(_at|_a_at|_s_at)$)", merged_data$ID_REF), ]

duplicated_probes <- merged_data_filtered$ID_REF[duplicated(merged_data_filtered$ID_REF)]

if (length(duplicated_probes) > 0) {
  cat("Sonde duplicate trovate (ID_REF):\n")
  print(duplicated_probes)
  # Aggiungi numerazione progressiva alle probes duplicate
  merged_data_filtered$ID_REF <- ave(merged_data_filtered$ID_REF, merged_data_filtered$ID_REF, 
                                     FUN = function(x) paste0(x, "_", seq_along(x)))
} else {
  cat("Non ci sono sonde duplicate (ID_REF).\n")
}

write_xlsx(merged_data_filtered, "GSE43488_mapped.xlsx")
