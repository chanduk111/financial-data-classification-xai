# Financial Data Classification with Explainable AI

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine-learning project for **retail financial data classification** using the Walmart M5 dataset. The project combines data preparation, feature engineering, NLP with TF-IDF, supervised classification and Explainable AI (XAI).

Developed as part of an **MSc Computer Science** project by **Kundeti Sai Chandra Sekhar**.

## Why this project?

Retail datasets contain a mixture of numerical, categorical, temporal and event information. This project explores how those signals can be transformed into machine-learning features and used to classify retail departments, while also providing interpretable explanations for model behaviour.

## Workflow

```mermaid
flowchart LR
    A[Walmart M5 Data] --> B[Data Cleaning & Integration]
    B --> C[Feature Engineering]
    C --> D[NLP / TF-IDF]
    D --> E[Train/Test Split]
    E --> F[Model Comparison]
    F --> G[Evaluation]
    G --> H[Explainable AI]
    H --> I[Business Insights]
```

## Technical stack

- **Python**
- **Pandas / NumPy**
- **scikit-learn**
- **Natural Language Processing (TF-IDF)**
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Gaussian Naive Bayes**
- **LIME**
- **SHAP**
- **Matplotlib**
- **WordCloud**
- **Joblib**
- **Jupyter / Google Colab**

## Implementation

### 1. Data preparation

The original workflow loads:

- `sales_train_validation.csv`
- `calendar.csv`
- `sell_prices.csv`

It selects the most recent 365 sales columns, creates a representative 1,000-product sample, reshapes sales data into long format, and joins calendar and selling-price information.

Missing selling prices are imputed with the median, remaining missing values are handled, and duplicate records are removed.

### 2. Feature engineering

The workflow derives:

- Year
- Month
- Week
- Day
- Day name
- Weekend indicator
- Price category

Categorical variables are label encoded before model training.

### 3. NLP

Event-related fields are combined into a text feature using:

- `event_name_1`
- `event_name_2`
- `event_type_1`
- `event_type_2`

The text is cleaned and transformed with **TF-IDF**, with the original experiment limiting the vocabulary to 15 features.

### 4. Machine learning

Four classifiers are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gaussian Naive Bayes

Evaluation uses weighted:

- Accuracy
- Precision
- Recall
- F1-score

The Random Forest configuration also uses 3-fold cross-validation on a sample of the training data.

## Results

The following results are the **reported results from the MSc experiment**:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 38.2% | 34.5% | 38.2% | 27.1% |
| **Decision Tree** | **74.7%** | **74.7%** | **74.7%** | **74.7%** |
| Random Forest | 57.6% | 57.6% | 57.6% | 57.6% |
| Gaussian Naive Bayes | 37.0% | 27.8% | 37.0% | 26.9% |

**Key result:** the Decision Tree produced the highest documented accuracy and F1-score at **74.7%**.

> The values above are preserved from the original MSc experiment. They are not presented as a new benchmark independently rerun from this repository.

## Explainable AI

### Global explainability

Random Forest feature importance is used to rank the contribution of engineered features to the model's classification behaviour.

The documented analysis identified features including **`sell_price`**, **`store_id`** and **`Price_Category`** among the strongest features.

### Local explainability

**LIME (Local Interpretable Model-Agnostic Explanations)** is used to explain an individual Random Forest prediction by showing feature contributions for a selected sample.

### SHAP

SHAP was included in the original project methodology as an Explainable AI technique. The repository preserves the project's XAI scope without claiming additional SHAP experiments that are not present in the supplied implementation.

## Model export and reproducibility

The original implementation demonstrates saving and reloading:

- Random Forest model
- Feature scaler
- Feature-name list

using **Joblib**, followed by validation against a sample of 100 test records.

The reusable implementation in `src/` separates preprocessing, modelling and explainability utilities.

## Repository structure

```text
financial-data-classification-xai/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── financial_data_classification.ipynb
├── results/
│   ├── model_comparison.csv
│   └── README.md
└── src/
    ├── __init__.py
    ├── preprocess.py
    ├── models.py
    ├── xai.py
    └── train.py
```

## Getting started

### 1. Clone

```bash
git clone https://github.com/chanduk111/financial-data-classification-xai.git
cd financial-data-classification-xai
```

### 2. Create an environment

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the three required Walmart M5 files in:

```text
data/raw/
```

See [`data/README.md`](data/README.md).
### 5. Run the notebook

The complete documented MSc experiment is contained in:

notebooks/financial_data_classification.ipynb

Open the notebook in Jupyter or Google Colab, update the dataset path if required, and run the cells in sequence. The notebook contains the dataset preparation, feature engineering, TF-IDF transformation, model training, evaluation and XAI analysis.

### 6. Reusable Python modules

The `src/` directory contains reusable preprocessing, modelling and explainability utilities. After constructing the feature matrix `X` and target `y` from the dataset workflow:

```python
from src.train import train_and_evaluate

results = train_and_evaluate(X, y)
print(results)
