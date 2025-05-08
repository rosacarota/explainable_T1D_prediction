library(readxl)
library(writexl)

file_path <- "threshold.xlsx"
dataset <- read_excel(file_path)

genes_data <- dataset[, -c(1, (ncol(dataset)-2):ncol(dataset))]

t_test_results <- apply(genes_data, 2, function(x) {
  t.test(x ~ dataset$Illness)
})

p_values <- sapply(t_test_results, function(test) test$p.value)

significant_genes <- names(p_values[p_values < 0.05])
print(significant_genes)

significant_p_values <- p_values[p_values < 0.05]
print(significant_p_values)

df <- data.frame(Gene = significant_genes, P_value = significant_p_values)
write_xlsx(df, "p-value.xlsx")

significant_genes_data <- genes_data[, significant_genes]

final_dataset <- cbind(dataset[, 1], significant_genes_data, dataset[, (ncol(dataset)-2):ncol(dataset)])

write_xlsx(final_dataset, "test.xlsx")

