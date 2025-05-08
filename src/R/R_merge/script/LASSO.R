library(glmnet)
library(readxl)
library(writexl)

dataset <- read_excel("threshold.xlsx")

# Escludi ID_REF, Age, Gender
x <- as.matrix(dataset[, -c(1, (ncol(dataset)-2):ncol(dataset))])
y <- ifelse(dataset$Illness == "Type 1 Diabetes", 1, 0)

lasso_model <- cv.glmnet(x, y, alpha = 1, family = "binomial")

best_lambda <- lasso_model$lambda.min

# Ottieni i coefficienti delle feature selezionate con il miglior lambda
lasso_coef <- as.matrix(coef(lasso_model, s = best_lambda))

# Filtra le feature (geni) che hanno coefficienti diversi da zero
selected_genes <- rownames(lasso_coef)[lasso_coef[, 1] != 0]


# Rimuovi "(Intercept)" dalla lista dei geni selezionati, se presente
selected_genes <- selected_genes[selected_genes != "(Intercept)"]

filtered_dataset <- dataset[, c("ID_REF", selected_genes, "Illness", "Age", "Gender")]

write_xlsx(filtered_dataset, "threshold.xlsx")
