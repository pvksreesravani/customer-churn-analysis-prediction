# Customer Churn Analysis & Prediction System

A beginner-friendly Python project that analyses customer behaviour and predicts which customers are likely to **churn** (stop buying). It uses a **synthetic dataset of 2,000 customers** generated in code, so no external datasets, APIs or file uploads are needed.

## Project Overview

The script (`churnprediction.py` / `churnprediction.ipynb`) does the following:

1. **Generates** a reproducible dataset of 2,000 customers (fixed seed `42`). Fewer purchases, more days since the last purchase and more support tickets raise churn probability.
2. **Cleans** deliberately messy data: missing values, duplicate CustomerIDs, invalid ranges and inconsistent churn labels.
3. **Analyses** overall churn rate, churn by age group and spending segment, and the top 3 factors linked to churn.
4. **Visualises** churn-rate bar charts, a correlation heatmap and a box plot of `LastPurchaseDaysAgo` by `Churned` (plus a confusion matrix and feature importance chart).
5. **Trains** a Random Forest classifier (80/20 stratified split, `random_state=42`) inside a scikit-learn Pipeline.
6. **Evaluates** with accuracy, precision, recall, F1-score, confusion matrix and classification report.
7. **Predicts** churn for a new customer with `predict_churn()` and `show_prediction()`, which validate inputs and return the status and churn probability.
8. **Saves** the cleaned data (`customer_churn_data.csv`) and the trained model (`churn_model.joblib`).

### Dataset columns

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier (not used as a model feature) |
| Age | Customer age in years |
| MembershipDuration | Months since the customer joined |
| AvgMonthlySpend | Average monthly spend |
| NumPurchasesLast3Months | Number of purchases in the last 3 months |
| SupportTicketsRaised | Number of support tickets raised |
| DiscountUsageRate | Share of purchases made with a discount (0 to 1) |
| LastPurchaseDaysAgo | Days since the last purchase |
| Churned | Target: 1 = churned, 0 = stayed |

### Data-leakage prevention
- `CustomerID` and `Churned` are excluded from the model features.
- The train/test split is done before anything is learned from the data.
- Missing values are filled by a median imputer inside the Pipeline, fitted on training data only.

## Installation

**Google Colab (recommended):** nothing to install. The script installs any missing package automatically.

**Local machine (optional):**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn joblib
python churnprediction.py
```

## How to Run

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook (or upload `churnprediction.ipynb`).
2. Paste the contents of `churnprediction.py` into a single code cell.
3. Press **Shift + Enter**.
4. Read the printed analysis, view the plots and check the model metrics.
5. Download `customer_churn_data.csv` and `churn_model.joblib` from the Colab **Files** panel.

### Predict for a new customer
```python
show_prediction(age=35, membership_duration=18, avg_monthly_spend=70,
                num_purchases_last_3_months=4, support_tickets_raised=2,
                discount_usage_rate=0.3, last_purchase_days_ago=60)
```
For typed prompts, set `USE_INTERACTIVE_INPUT = True` at the bottom of the script and re-run.

### Reuse the saved model
```python
import joblib
import pandas as pd

model = joblib.load("churn_model.joblib")
customer = pd.DataFrame([[35, 18, 70.0, 4, 2, 0.3, 60]],
                        columns=["Age", "MembershipDuration", "AvgMonthlySpend",
                                 "NumPurchasesLast3Months", "SupportTicketsRaised",
                                 "DiscountUsageRate", "LastPurchaseDaysAgo"])
print(model.predict_proba(customer)[0][1])   # probability of churn
```

## Key Insights

### Data cleaning
- Raw dataset: 2,030 rows. Cleaned dataset: **2,000 rows** with no missing values.
- 25 duplicate CustomerIDs removed, 5 rows with unusable churn labels dropped, and 40 invalid values (impossible ages, negative spend and similar) corrected.

### Churn analysis
- **Overall churn rate: 30.0%** (601 of 2,000 customers).
- **By spending segment:** Low spenders churn at **41.4%**, Medium at 31.0% and High at **17.7%**. Spending is the clearest segment difference.
- **By age group:** churn is fairly similar across ages (26.5% to 31.1%), so age is a weak predictor. The 56+ group is lowest (26.5%) and the 36-45 and 46-55 groups are highest (31.1%).

### Top 3 factors associated with churn

| Rank | Factor | Correlation with churn | Meaning |
|---|---|---|---|
| 1 | NumPurchasesLast3Months | -0.390 | More purchases means less churn |
| 2 | LastPurchaseDaysAgo | +0.382 | A longer gap since the last purchase means more churn |
| 3 | AvgMonthlySpend | -0.207 | Higher spend means less churn |

`SupportTicketsRaised` is close behind (+0.194): more support tickets means more churn.

### Model performance (Random Forest, 400-customer test set)

| Metric | Score |
|---|---|
| Accuracy | 0.757 |
| Precision (churn class) | 0.573 |
| Recall (churn class) | 0.750 |
| F1-score (churn class) | 0.650 |

**Confusion matrix**

| | Predicted Stayed | Predicted Churned |
|---|---|---|
| **Actual Stayed** | 213 | 67 |
| **Actual Churned** | 30 | 90 |

The model catches **75% of customers who actually churn** (90 of 120). Class weights are balanced, which favours recall. The trade-off is lower precision: some loyal customers are flagged as at-risk, which is acceptable when a retention offer is cheap.

### Example predictions
- Disengaged customer (1 purchase, 150 days inactive, 5 tickets): **Churn, 96.3% probability, High risk**.
- Loyal customer (12 purchases, 5 days since last purchase, 0 tickets): **Stay, 1.5% probability, Low risk**.

### Business recommendations
- Run **win-back campaigns** for customers who have not purchased in a long time.
- Offer **targeted retention deals** to Low-spend customers, who churn most.
- **Improve support** for customers raising multiple tickets.
- Reward and engage **frequent buyers** to keep their purchase frequency high.

### Limitations
- The data is **synthetic**, so these results show that the method works. They are not real business findings.
- The model should be re-validated and tuned on real customer data before any real-world use.

## Project Structure

```
churn-prediction/
├── churnprediction.py          # full Python script
├── churnprediction(1).ipynb       # Google Colab notebook with outputs
├── customer_churn_data.csv     # cleaned dataset (generated)
├── churn_model(1).joblib          # trained Random Forest pipeline (generated)
└── README.md
```

## 📊 Project Visualizations

### 1. Churn Rate Analysis
![Churn Rate Analysis](<Output Images/Churnrate Analysis.png>)

### 2. Random Forest Confusion Matrix
![Random Forest Confusion Matrix](<Output Images/RandomForest ConfusionMatrix.png>)

### 3. Correlation Heatmap
![Correlation Heatmap](<Output Images/correlationheatmap.png>)

### 4. Last Purchase Days vs Churn
![Last Purchase Days vs Churn](<Output Images/lastpurchase vs churn.png>)

## Technologies Used
Python, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Joblib

## License
For educational use as part of an internship project.
