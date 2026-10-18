# Customer Churn Analysis & Prediction System

## Project Overview
This project analyses customer behaviour and predicts which customers are likely
to **churn** (stop buying). It uses a **synthetic dataset of 2,000 customers**
generated in code, so no external data, APIs or file uploads are needed.

**What the script does**
1. Generates 2,000 customers with realistic churn patterns (fewer purchases,
   longer time since last purchase and more support tickets raise churn risk).
2. Cleans messy data: missing values, duplicate CustomerIDs, invalid ranges and
   inconsistent labels.
3. Analyses overall churn rate, churn by age group and spending segment, and
   the top 3 factors linked to churn.
4. Draws plots: churn-rate bar charts, correlation heatmap, box plot of
   `LastPurchaseDaysAgo` by `Churned` (plus confusion matrix and feature importance).
5. Trains a Random Forest (80/20 split, `random_state=42`) inside a scikit-learn
   Pipeline, and reports accuracy, precision, recall, F1, confusion matrix and
   classification report.
6. Provides `predict_churn()` with input validation, plus an optional
   interactive version.
7. Saves `customer_churn_data.csv` and `churn_model.joblib`.

**Dataset columns:** CustomerID, Age, MembershipDuration, AvgMonthlySpend,
NumPurchasesLast3Months, SupportTicketsRaised, DiscountUsageRate,
LastPurchaseDaysAgo, Churned.

**Data-leakage prevention:** `CustomerID` and `Churned` are excluded from the
features; the train/test split happens before preprocessing is learned; missing
values in the model are filled by an imputer fitted on training data only.

## Installation
**Google Colab (recommended):** nothing to install. The script installs any
missing package automatically.

**Local machine (optional):**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## How to Run
1. Open https://colab.research.google.com and create a **New notebook**.
2. Paste the complete script into a single code cell.
3. Press **Shift + Enter** (or the Run button).
4. Wait for the output: tables, 3+ plots, model metrics and prediction examples.
5. Download `customer_churn_data.csv` and `churn_model.joblib` from the
   **Files** panel on the left (right-click the file, then Download).

To predict for your own customer, call:
```python
show_prediction(age=35, membership_duration=18, avg_monthly_spend=70,
                num_purchases_last_3_months=4, support_tickets_raised=2,
                discount_usage_rate=0.3, last_purchase_days_ago=60)
```
For typed prompts, set `USE_INTERACTIVE_INPUT = True` at the bottom and re-run.

To reuse the saved model later:
```python
import joblib
model = joblib.load("churn_model.joblib")
```

## Key Insights to Report
Copy the real numbers from your own run into these points:
- **Overall churn rate:** X% of customers churned.
- **Segments:** which age group and which spending segment (Low/Medium/High)
  churn most, and how far they are above or below the overall rate.
- **Top 3 churn factors:** the three features with the strongest correlation
  with churn and their direction (expected: `LastPurchaseDaysAgo` and
  `SupportTicketsRaised` up = more churn; `NumPurchasesLast3Months` up = less churn).
- **Model performance:** accuracy, precision, recall and F1 on the test set.
  Explain what recall means for the business (how many real churners are caught).
- **Business recommendations:** e.g. run win-back campaigns for customers
  inactive for a long time, improve support for customers with many tickets,
  and reward frequent buyers.
- **Limitations:** the data is synthetic, so results show the method works and
  are not real business findings. Real data would need further validation.

## Project Structure
```
churn-prediction/
├── churn_analysis.py            # the full script (or a Colab notebook)
├── customer_churn_data.csv      # cleaned dataset (generated)
├── churn_model.joblib           # trained model (generated)
└── README.md
```

## GitHub Upload Instructions
**Option A: Command line**
```bash
# 1. Create an empty repository on github.com (e.g. "churn-prediction"), no README
# 2. In your project folder:
git init
git add churn_analysis.py customer_churn_data.csv churn_model.joblib README.md
git commit -m "Add customer churn analysis and prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/churn-prediction.git
git push -u origin main
```
**Option B: Web upload (no Git needed)**
1. Create a new repository on github.com.
2. Click **Add file -> Upload files**, drag in your files, then click **Commit changes**.

**Option C: Directly from Colab**
Use **File -> Save a copy in GitHub**, choose your repository and commit.

Tip: save the script as `churn_analysis.py` (or download the notebook as `.ipynb`)
and add a short project description on the repository page.

## License
For educational use as part of an internship project.