
# Explainable Machine Learning per la Classificazione del Diabete di Tipo 1 (T1D)

> **Tesi di Laurea Triennale**  
> Corso di Laurea in Informatica – Università degli Studi di Salerno  
> **Autrice**: Rosa Carota

## Obiettivi
- Classificare soggetti sani e affetti da T1D utilizzando modelli di machine learning.
- Applicare tecniche XAI per interpretare le predizioni.
- Valutare le performance dei modelli tramite metriche standard.
- Analizzare l’importanza delle feature nei modelli.

## Tecnologie Utilizzate
- **Linguaggi**: Python, R
- **Python**:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`
  - `shap`
  - `alibi`
- **Explainability Tools**:
  - SHAP (TreeExplainer)
  - Anchor (AnchorTabular)
- **R**:
  - Utilizzato per preprocessing e mappatura dei geniù

## Modelli e Tecniche
- **Support Vector Machine (SVM)** – Miglior accuratezza tra i modelli testati
- **Random Forest**, **XGBoost** – Utilizzati per confronto
- **SHAP** – Interpretazione globale e locale dell'importanza delle feature
- **Anchor** – Spiegazioni locali basate su regole

## Come Iniziare

1. **Clona la repository**:
```bash
git clone https://github.com/rosacarota/explainable_T1D_prediction.git
cd explainable_T1D_prediction
```
2. **Clona la repository**:
- Installare le dipendenze di Python:
```bash
pip install -r requirements.txt
```
- Installare i pacchetti necessari indicati negli script presenti nella cartella src/.
3. **Clona la repository**:

## Contatti
Rosa Carotenuto - r.carotenuto16@studenti.unisa.it
