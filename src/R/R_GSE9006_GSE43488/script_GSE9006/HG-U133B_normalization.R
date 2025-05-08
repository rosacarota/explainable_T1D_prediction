library(affy)
library(openxlsx)

cel_path <- "CEL\\piattaforma_1"

cel_files <- list.files(cel_path, pattern = "\\.CEL$", full.names = TRUE)

if(length(cel_files) == 0) {
  stop("Nessun file CEL trovato nel percorso specificato.")
}

data <- ReadAffy(filenames = cel_files)

# Applica la normalizzazione RMA
eset <- rma(data)

# Estrai le espressioni normalizzate
exprs_data <- exprs(eset)

# Estrai i nomi delle sonde
probe_names <- featureNames(eset)

exprs_df <- as.data.frame(exprs_data)
exprs_df$Probe_ID <- probe_names

exprs_df <- exprs_df[, c(ncol(exprs_df), 1:(ncol(exprs_df) - 1))]

output_file <- "HG-U133A_normalized.xlsx"
write.xlsx(as.data.frame(exprs_df), file = output_file)
