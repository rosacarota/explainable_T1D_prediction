library(affy)
library(affyio)
library(openxlsx)

cel_path = "piattaforma_1"

txt_file_path = "pazienti_healthy.txt"

list_of_names <- readLines(txt_file_path)
list_of_names <- paste0(list_of_names, ".CEL")
print(list_of_names)

cel_files <- as.character(list.files(cel_path, pattern = "\\.CEL$", full.names = TRUE))
print(cel_files)
filtered_cel_files <- cel_files[basename(cel_files) %in% list_of_names]
print(filtered_cel_files)
data = ReadAffy(filenames = filtered_cel_files)
print(data)


cel_path_2 = "CEL\\piattaforma_2"

cel_files_2 <- as.character(list.files(cel_path_2, pattern = "\\.CEL$", full.names = TRUE))
print(cel_files_2)

if (!is.character(cel_files_2)) {
  stop("Errore: cel_files non è un vettore di caratteri")
}

if(length(cel_files_2) == 0) {
  stop("Nessun file CEL trovato nel percorso specificato.")
}

data_2 = ReadAffy(filenames = cel_files_2)
print(data_2)
