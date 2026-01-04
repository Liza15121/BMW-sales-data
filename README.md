# BMW sales data

**Sales Conversion Prediction**
    Predict whether a sales lead will convert into a high sale or low sale using machine learning classification models.

**Problem statement**
    Many businesses collect large volumes of lead and transaction data but struggle to identify which leads are most likely to convert.

    This project builds a classification model that predicts the probability of conversion so sales teams can prioritize high‑value opportunities.

---------------------------------------------------------------------------------

***Project Structure***
.
├── data/
│   ├── raw/               # Original dataset (read-only)
│   ├── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
├── src/
│   ├── data/              # Data loading and preprocessing scripts
│   ├── features/          # Feature engineering functions
│   ├── models/            # Training and inference code
│   └── utils/             # Helper utilities and configuration
├── models/                # Serialized trained models
├── requirements.txt
├── README.md
└── config.yaml

---------------------------------------------------------------------------------

**Methodology**
*1. Problem framing*
•	Task: Binary classification (high sale vs low sale).
•	Objective: Maximize business impact (e.g., higher conversion rate for contacted leads) rather than accuracy alone.

*2. Data preparation*
•   Handle missing values, detect outliers, and correct data types.
•	Encode categorical variables (one‑hot/target/ordinal encoding) and scale numeric features where needed.

*3. Feature engineering*
•	Create recency, frequency, and monetary (RFM‑style) features.
•	Aggregate historical interactions per customer or account.
•	Derive interaction‑level ratios (discount %, contact rate, response rate).

*4. Modeling*
•	Baseline: Logistic Regression.
•	Tree‑based models: Random Forest, Gradient Boosting (XGBoost/LightGBM).
•	Hyperparameter tuning via cross‑validation (GridSearch/RandomizedSearch).

*5. Evaluation metric* 
•	Main metrics: ROC‑AUC, precision, recall, F1‑score, PR‑AUC.
•	Business metrics: uplift in conversion for top‑N leads, cost–benefit analysis of contacting leads.
•	Analyze confusion matrix and calibration curves to understand trade‑offs.

---------------------------------------------------------------------------------

# **Installations**
# Clone the repository
git clone https://github.com/<your-username>/sales-conversion-prediction.git
cd sales-conversion-prediction

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# **Usage**
- Run EDA 
Open `notebooks/01_eda.ipynb` to explore data distributions, target leakage, and class imbalance

- Train model from CLI(example):
python -m src.models.predict \
  --model-path models/sales_classifier.pkl \
  --input data/processed/new_leads.csv \
  --output predictions.csv

- Make predictions(example:
python -m src.models.predict \
  --model-path models/sales_classifier.pkl \
  --input data/processed/new_leads.csv \
  --output predictions.csv

These commands can be adapted for running inside a notebook or pipeline.

---------------------------------------------------------------------------------

# Results
- [ ] Best Model: ____________ (e.g: Gradient Boosting with tuned hyperparameters)
- [ ] Performance: ___________ (e.g: replace with actual numbers)
    •	ROC‑AUC: ___________ on test set.
    •	Precision@top‑20% leads: _________ vs ________ baseline.
- [ ] Key drivers of conversion (feature importance/SHAP):
    •	Recent activity, number of interactions, discount level, product category, and customer segment.

---------------------------------------------------------------------------------

**Handling class imbalance**
Sales conversion datasets are often highly imbalanced (few positives).

- Strategies used: 
    - Class weights in algorithms like Logistic Regression and tree‑based models.
    - Oversampling (SMOTE) or undersampling on the training set only.

---------------------------------------------------------------------------------

# Future work
- Deploy model as a FLASK API or Streamlit dashboard for interactive scoring.
- Experiment with more advanced models (CatBoost, stacking, or time‑aware models if data is temporal).

---------------------------------------------------------------------------------



