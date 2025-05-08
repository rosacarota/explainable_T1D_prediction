
library(readxl)
library(writexl)

dataset <- read_excel("batch.xlsx")

str(dataset)

threshold <- 5

expression_data <- dataset[, -c(1, (ncol(dataset)-2):ncol(dataset))]

genes_to_keep <- apply(expression_data, 2, function(x) any(x >= threshold))

filtered_expression_data <- expression_data[, genes_to_keep]

dataset_final <- cbind(dataset[, 1], filtered_expression_data, dataset[, (ncol(dataset)-2):ncol(dataset)])

write_xlsx(dataset_final, "threshold.xlsx")
