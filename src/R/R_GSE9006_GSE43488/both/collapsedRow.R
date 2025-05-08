library(WGCNA)
library(readxl)
install.packages("writexl")
library(writexl)

gene_data <- read_excel("HG-U133A_mapped_without_NA.xlsx")

probe_names <- as.character(gene_data$Probe_ID)
print(probe_names)

gene_names <- as.character(gene_data$SYMBOL)

expression_data <- gene_data[, -which(names(gene_data) %in% c("SYMBOL", "GENENAME"))]

expression_data <- as.data.frame(expression_data)

rownames(expression_data) <- expression_data$Probe_ID

expression_data$Probe_ID <- NULL

str(expression_data)

if (nrow(expression_data) != length(gene_names) || nrow(expression_data) != length(probe_names)) {
  stop("Il numero di righe in expression_data non corrisponde alla lunghezza di gene_names o probe_names")
}

is_numeric <- sapply(expression_data, function(col) all(is.numeric(col)))

print(is_numeric)

if (!all(is_numeric)) {
  cat("Attenzione: ci sono colonne non completamente numeriche.\n")
  
  non_numeric_cols <- names(expression_data)[!is_numeric]
  print(paste("Colonne non numeriche:", non_numeric_cols))
}
  
duplicate_probes <- probe_names[duplicated(probe_names)]
cat("Probes duplicate trovate:\n")
print(duplicate_probes)

collapsed_data <- collapseRows(
  datET = as.data.frame(expression_data),
  rowGroup = gene_names,
  rowID = probe_names,
  method = "MaxMean",
  connectivityBasedCollapsing = FALSE
)

collapsed_data_clean <- collapsed_data$datETcollapsed
print(head(collapsed_data))

collapsed_data_clean <- as.data.frame(collapsed_data_clean)

collapsed_data_clean_with_row_names <- cbind(ID_REF = rownames(collapsed_data_clean), collapsed_data_clean)

write_xlsx(collapsed_data_clean_with_row_names, "collapsed_data_HG-U133A.xlsx")
