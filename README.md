# ExtraaLearn Lead Conversion — Prediction, Deployment & Business Insights

## 1. Project Title

**ExtraaLearn Lead Conversion Prediction: Model Development and Deployment**

## 2. Business Problem

ExtraaLearn is an early-stage EdTech startup offering programs on cutting-edge technologies to help students and professionals upskill/reskill. Like most EdTech companies, ExtraaLearn generates a large volume of leads through its website, mobile app, digital advertising, referrals, and other channels. Not every lead converts into a paying customer, and the sales team has limited capacity to follow up with all of them. The company needs a data-driven way to identify which leads are most likely to convert so outreach effort can be prioritized effectively.

## 3. Objective

Using the leads dataset provided by ExtraaLearn, this project:
- Builds a machine learning model to predict which leads are likely to convert to paid customers.
- Identifies the factors most associated with lead conversion.
- Produces a profile of leads that are more likely to convert.
- Deploys the trained model as a usable web service (Flask API + Streamlit UI) so the sales team could act on predictions.

## 4. Dataset Overview

- **Source file:** `ExtraaLearn.csv`
- **Shape:** 4,612 rows × 15 columns
- **Missing values:** none (confirmed via `df.isnull().sum()` — every column is 100% complete)
- **Duplicate rows:** none (`df.duplicated().sum() == 0`)
- **Duplicate IDs:** none — all 4,612 `ID` values are unique
- **Columns:** `ID`, `age`, `current_occupation`, `first_interaction`, `profile_completed`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`, `last_activity`, `print_media_type1`, `print_media_type2`, `digital_media`, `educational_channels`, `referral`, `status`

## 5. Target Variable

`status` — already a binary numeric target:
- `0` = Not Converted (3,235 leads, 70.14%)
- `1` = Converted (1,377 leads, 29.86%)

**Overall conversion rate: 29.86%.** No remapping of the target was needed. The class imbalance (~70/30) directly informed the choice of evaluation metric (see Section 11).

## 6. Exploratory Data Analysis

EDA covered both univariate and bivariate analysis:
- **Univariate:** distribution/boxplots for `age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`; count plots for all categorical fields (`current_occupation`, `first_interaction`, `profile_completed`, `last_activity`, and the five marketing-exposure flags).
- **Bivariate:** each numeric variable's distribution and boxplot split by `status`, plus conversion-rate tables and stacked bar charts for every categorical variable against `status`.

## 7. Key EDA Findings

All figures below come directly from the executed analysis (see the notebook for the full breakdown):

- Converted leads spend **much more time on the website** — median 789s vs. 317s for non-converted leads (mean 1,068s vs. 577s), the clearest numeric behavioral difference found.
- **First interaction channel** shows a large conversion gap: **Website 45.59%** vs. **Mobile App 10.53%**.
- **Profile completion** shows a clean, monotonic relationship: **High 41.78% > Medium 18.88% > Low 7.48%**.
- **Occupation** matters: **Professional 35.51% > Unemployed 26.58% > Student 11.71%**.
- **Last activity** differentiates leads: **Website Activity converts at 38.45%**, higher than Email Activity (30.33%) or Phone Activity (21.31%), even though Email Activity is the most *common* last touchpoint.
- **Referral** shows the largest single conversion-rate gap of any variable — **67.74%** conversion vs. a 29.08% baseline for non-referred leads — but referrals are a very small group (**only 93 of 4,612 leads**), so this figure should be **interpreted cautiously** given the small sample size.
- `website_visits` and `page_views_per_visit` show little difference between converted and non-converted leads on their own — engagement *duration* (time spent) is a stronger raw signal than visit *frequency*.

These are **associations observed in the data, not causal claims.**

## 8. Data Preprocessing

- **Target:** `status` (already binary, 0/1 — no remapping needed).
- **Features (`X`):** all columns except `ID` and `status` (13 predictor columns).
- **`ID` excluded from modeling** — it is a unique, non-informative row identifier (verified: 4,612 unique values for 4,612 rows) with no predictive meaning.
- **Feature engineering:** none added. The raw variables already carried the key EDA-validated signals, and combining the marketing-exposure flags was considered but rejected, since it would have blurred `referral`'s distinct signal from the other four (weaker) flags.
- **Train/test split:** stratified 80/20 split (`test_size=0.20, random_state=42, stratify=y`), performed **before** any preprocessing was fit, to keep the test set genuinely held out. Stratification was used because the target is moderately imbalanced (~70/30), ensuring both splits preserve that balance.
  - Training set: 3,689 rows
  - Test set: 923 rows
- **Preprocessing pipeline** (`ColumnTransformer`, fit only on training data):
  - Numerical features (`age`, `website_visits`, `time_spent_on_website`, `page_views_per_visit`): `SimpleImputer(strategy="median")` + `StandardScaler` (scaling included for pipeline generality; not required by the tree-based models actually used).
  - Categorical features (`current_occupation`, `first_interaction`, `profile_completed`, `last_activity`, and the 5 marketing-exposure flags): `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
- **Leakage prevention:** the preprocessing pipeline is embedded inside every model `Pipeline`, so it is refit independently within each cross-validation fold and only ever fit on `X_train`; `X_test` is only ever `.transform()`-ed, never used to fit anything.

## 9. Outlier Treatment

IQR-based analysis on the four numeric features:

| Feature | Q1 | Q3 | Upper Bound | Outliers |
|---|---|---|---|---|
| age | 36.0 | 57.0 | 88.5 | 0 (0.00%) |
| website_visits | 2.0 | 5.0 | 9.5 | 154 (3.34%) |
| time_spent_on_website | 148.75 | 1336.75 | 3118.75 | 0 (0.00%) |
| page_views_per_visit | 2.08 | 3.76 | 6.27 | 257 (5.57%) |

**Decision: outliers were retained, not capped or removed.** The flagged points in `website_visits` and `page_views_per_visit` represent plausible, highly engaged lead behavior rather than data errors (no negative values, no implausible ranges found anywhere in the dataset). Both models used in this project (Random Forest, AdaBoost) are **tree-based ensembles that split on threshold values**, so they are inherently robust to extreme values in a single feature — aggressive outlier treatment was judged unnecessary and potentially harmful to genuine signal (since `time_spent_on_website`, which showed 0% outliers, was already known from EDA to be a strong conversion signal).

## 10. Modeling Approach

Two tree-based ensemble classifiers were built and compared, each inside a complete `Pipeline` with the shared preprocessing `ColumnTransformer`:
1. **Random Forest** (`RandomForestClassifier`)
2. **AdaBoost** (`AdaBoostClassifier`)

Both were first evaluated as reproducible baselines (`random_state=42`), then improved via hyperparameter tuning.

## 11. Evaluation Metric

**Primary metric: Recall (of the "Converted" / positive class).**

Rationale: the target is imbalanced (~70% not converted / ~30% converted), so accuracy alone would be misleading — a model predicting "not converted" for every lead would already score ~70% accuracy while being useless to the business. Considering the cost of each error type:
- **False Negative** (missing an actual converter): a lost revenue opportunity — the sales team never follows up with a real paying customer.
- **False Positive** (flagging a non-converter as likely): a wasted-outreach cost, comparatively cheaper than losing a genuine customer.

Since missing a genuine converter is more costly than spending some extra effort on a lead who doesn't convert, **Recall** was used consistently as the primary metric for comparing and selecting between all model variants. **Precision, F1, Accuracy, and ROC-AUC** were tracked throughout as secondary metrics.

## 12. Baseline Model Results

| Model | CV Recall | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (Baseline) | 0.7184 | 0.8570 | 0.7857 | 0.7174 | 0.7500 | 0.9166 |
| AdaBoost (Baseline) | 0.6322 | 0.8321 | 0.7318 | 0.6920 | 0.7114 | 0.9005 |

Random Forest outperformed AdaBoost on every metric at baseline, though it showed clear training-set overfitting (train Recall 1.0 vs. test Recall 0.7174), while AdaBoost showed almost no train/test gap but a lower overall performance ceiling.

## 13. Hyperparameter Tuning

- **Random Forest:** tuned via `RandomizedSearchCV` (25 of 144 possible combinations sampled, `scoring="recall"`, 5-fold `StratifiedKFold(shuffle=True, random_state=42)`) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`.
- **AdaBoost:** tuned via exhaustive `GridSearchCV` (12 combinations) over `n_estimators` and `learning_rate`, same CV setup.
- The test set was **not** used during either search — only touched once, after the best estimator was already selected.

| Model | CV Recall | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest (Tuned) | 0.7311 | 0.8613 | 0.7960 | 0.7210 | 0.7567 | 0.9232 |
| AdaBoost (Tuned) | 0.6422 | 0.8321 | 0.7318 | 0.6920 | 0.7114 | 0.9008 |

Tuning improved Random Forest across every metric and meaningfully reduced its overfitting gap (train Recall dropped from a perfect 1.0 to 0.9337, much closer to its test Recall of 0.7210). AdaBoost's tuned test metrics were essentially unchanged from baseline.

## 14. Final Model Selection

**Selected model: Random Forest (Tuned).** It has the highest CV Recall and test Recall of all four model variants, and also leads on every secondary metric (Precision, F1, ROC-AUC), with a reduced overfitting gap compared to its own baseline. AdaBoost, tuned or not, did not match Random Forest on any metric.

**Final model configuration:**
```
n_estimators=100
max_depth=None
min_samples_split=5
min_samples_leaf=1
max_features="sqrt"
bootstrap=True
criterion="gini"
random_state=42
```

## 15. Final Test Performance

Evaluated once on the untouched test set (923 leads):

| Metric | Value |
|---|---|
| Accuracy | 0.8613 |
| Precision | 0.7960 |
| Recall | 0.7210 |
| F1 | 0.7567 |
| ROC-AUC | 0.9232 |

## 16. Feature Importance / Key Conversion Drivers

Extracted from the fitted Random Forest inside the final pipeline (transformed feature names mapped back correctly, including one-hot encoded categories):

| Rank | Feature | Importance |
|---|---|---:|
| 1 | time_spent_on_website | 0.2516 |
| 2 | first_interaction_Website | 0.1178 |
| 3 | age | 0.0904 |
| 4 | page_views_per_visit | 0.0901 |
| 5 | first_interaction_Mobile App | 0.0853 |
| 6 | profile_completed_High | 0.0701 |
| 7 | website_visits | 0.0536 |
| 8 | profile_completed_Medium | 0.0477 |
| 9 | last_activity_Phone Activity | 0.0319 |
| 10 | current_occupation_Professional | 0.0303 |

These are **model-based importance scores** — they reflect how much each feature reduced impurity across the Random Forest's trees, and should be read as **predictive signals the model found useful, not causal drivers** of conversion.

## 17. High-Probability Lead Profile

Based on the combined EDA and feature-importance evidence (a descriptive profile, **not a guarantee** of conversion):

- **High website engagement** — especially a longer time spent on the site, and to a lesser extent more page views per visit.
- **First interaction via the Website** rather than the Mobile App.
- **Older leads** (skewing toward the upper end of the 18–63 age range in the dataset).
- **High or Medium profile completion.**
- **Professional occupation**, more so than Unemployed or Student leads.
- **Last activity recorded as Website Activity or Phone Activity.**
- A **referral source**, though this signal comes from a very small sample (93 leads) and should be weighted cautiously.

## 18. Business Recommendations

All tied directly to the validated findings above:

1. **Prioritize leads with stronger website engagement**, particularly time spent on site — the single strongest signal found.
2. **Prioritize leads with high profile completion** — encourage profile completion during onboarding, since it is both a strong predictor and a lever the business can influence.
3. **Pay attention to first-interaction channel** — Website-origin leads convert at over 4x the rate of Mobile App leads; consider whether the Mobile App onboarding/engagement experience can be improved.
4. **Use recent activity and engagement signals** (last activity type, time on site) for day-to-day lead prioritization, not just static demographic fields.
5. **Treat referral as a potentially valuable but small-sample signal** — its high conversion rate (67.74%) is promising but based on only 93 leads; it should inform, not solely drive, prioritization decisions.
6. **Use model predictions to prioritize sales effort, not to automatically reject lower-probability leads** — the model is a resource-allocation aid, not a gatekeeper. Lower-probability leads can still convert.

## 19. Project Structure

```
ExtraaLearn-Lead-Conversion/
│
├── notebook/
│   └── ExtraaLearn_Model_Deployment_Additional_Project_Full_Code.ipynb
│
├── data/
│   └── ExtraaLearn.csv
│
├── models/
│   └── extraalearn_lead_conversion_model.joblib
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/
│       └── extraalearn_lead_conversion_model.joblib
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── README.md
└── .gitignore
```

> Note: the notebook file keeps its original working filename (`ExtraaLearn_Model_Deployment_Additional_Project_Full_Code.ipynb`) rather than being renamed, so it stays consistent with the file actually developed and executed throughout this project.

## 20. How to Run the Notebook

```bash
cd notebook
pip install numpy==2.0.2 pandas==2.2.2 scikit-learn==1.6.1 matplotlib==3.10.0 seaborn==0.13.2 joblib==1.4.2 xgboost==2.1.4 requests==2.32.3 huggingface_hub==0.30.1
jupyter notebook ExtraaLearn_Model_Deployment_Additional_Project_Full_Code.ipynb
```
Run all cells sequentially from the top. The notebook expects `ExtraaLearn.csv` to be in its own working directory (or adjust the `pd.read_csv(...)` path to point at `../data/ExtraaLearn.csv`).

*(Note: the notebook was actually developed and validated in an environment with newer package versions — `numpy 2.4.4`, `pandas 3.0.2`, `scikit-learn 1.8.0`, `joblib 1.5.3` — which is why the deployment `requirements.txt` files below are pinned to those versions rather than the notebook's own install cell. Either version set will run the notebook; the deployment files must match whatever environment actually produced the serialized model.)*

## 21. Flask Backend

- **Location:** `backend/app.py`
- **Purpose:** loads the complete serialized pipeline (`backend/models/extraalearn_lead_conversion_model.joblib`) **once** at startup and serves predictions — never retrains or reloads per request.
- **Endpoints:**
  - `GET /` — health/root check.
  - `POST /predict` — accepts the 13 raw lead fields as JSON, returns `prediction`, `prediction_label`, and `conversion_probability`.
- **Error handling:** missing fields or non-numeric values in numeric fields → HTTP 400 with a descriptive message; unexpected exceptions → HTTP 500 with a generic message (no internal stack traces exposed). Unrecognized categorical values are intentionally accepted (not rejected), since the pipeline's `OneHotEncoder(handle_unknown="ignore")` already handles them safely.

**Local run instructions:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```
The API will be available at `http://localhost:5000`.

## 22. Streamlit Frontend

- **Location:** `frontend/app.py`
- **Purpose:** a pure UI layer — collects the 13 raw lead fields via form widgets and calls the Flask `/predict` endpoint. **It does not load the model itself** (no `scikit-learn`/`joblib` dependency), keeping the backend as the single source of inference truth.
- **Backend connection:** `BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000/predict")` — defaults to the local Flask server; configurable via environment variable. **No production/cloud URL is hardcoded anywhere.**

**Local run instructions:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```
The UI will be available at `http://localhost:8501` and will call the Flask API at `http://localhost:5000/predict` by default.

## 23. API Endpoint Documentation

### `GET /`
Health check. Returns:
```json
{"message": "ExtraaLearn Lead Conversion Prediction API is running", "status": "healthy"}
```

### `POST /predict`
Accepts a JSON object with the following 13 fields (exact names/values from `ExtraaLearn.csv`):

| Field | Type | Example values |
|---|---|---|
| age | integer | 18–63 |
| current_occupation | string | "Professional", "Unemployed", "Student" |
| first_interaction | string | "Website", "Mobile App" |
| profile_completed | string | "High", "Medium", "Low" |
| website_visits | integer | 0–30 |
| time_spent_on_website | integer | seconds, 0–2537 |
| page_views_per_visit | float | 0.0–18.434 |
| last_activity | string | "Email Activity", "Phone Activity", "Website Activity" |
| print_media_type1 | string | "Yes", "No" |
| print_media_type2 | string | "Yes", "No" |
| digital_media | string | "Yes", "No" |
| educational_channels | string | "Yes", "No" |
| referral | string | "Yes", "No" |

## 24. Example API Request

```json
POST /predict
{
    "age": 44,
    "current_occupation": "Professional",
    "first_interaction": "Mobile App",
    "profile_completed": "High",
    "website_visits": 6,
    "time_spent_on_website": 42,
    "page_views_per_visit": 3.395,
    "last_activity": "Website Activity",
    "print_media_type1": "No",
    "print_media_type2": "No",
    "digital_media": "No",
    "educational_channels": "No",
    "referral": "No"
}
```

## 25. Example API Response

```json
{
    "prediction": 0,
    "prediction_label": "Not Converted",
    "conversion_probability": 0.005
}
```

*(This is an actual response captured during local validation — see Section 28 for the full end-to-end test confirmation, including a positive-prediction example.)*

## 26. Docker Instructions

**Backend** (`python:3.11-slim`, Gunicorn, port 5000, bound to `0.0.0.0`):
```bash
docker build -t extraalearn-backend ./backend
docker run -p 5000:5000 extraalearn-backend
```

**Frontend** (`python:3.11-slim`, Streamlit, port 8501, bound to `0.0.0.0`):
```bash
docker build -t extraalearn-frontend ./frontend
docker run -p 8501:8501 -e BACKEND_URL=http://host.docker.internal:5000/predict extraalearn-frontend
```

> **Note:** these Docker commands have been documented but **not executed** in this project's development environment (no Docker daemon was available there). The Flask backend and Streamlit frontend were instead validated by running them directly as local Python/Streamlit processes (see Sections 21–22 and Section 28) — the Docker build/run steps above should be verified in an environment with Docker installed before relying on them for deployment.

## 27. Environment Variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `BACKEND_URL` | frontend/app.py | `http://localhost:5000/predict` | Where the Streamlit app sends prediction requests. Override for any non-local backend deployment. |
| `PORT` | backend/app.py | `5000` | Port the Flask dev server binds to when run directly with `python app.py` (Docker/Gunicorn deployment uses port 5000 explicitly in the `CMD`). |

No `.env` file, secrets, or credentials are required or included in this project.

## 28. Limitations

- The model's Recall (0.7210) means roughly **28% of actual converters are still missed** at the default 0.5 classification threshold — the model is a prioritization aid, not a perfect filter.
- The `referral` feature's strong raw conversion-rate association (67.74%) is based on only 93 leads and did not even appear in the model's top-15 feature importances — likely due to its small sample size limiting how much the model can learn from it. This finding needs more data to confirm.
- Model importances describe **association, not causation** — they should not be used to justify unilateral changes to marketing spend without further testing (e.g., A/B tests).
- Local validation (Sections 21–22) confirmed the Flask API and Streamlit UI work correctly and that Flask's response for a real dataset row matches direct model inference exactly — but the project has **not been deployed to any cloud provider**, and Docker builds have **not been executed** in this environment.
- The dataset reflects ExtraaLearn's historical lead behavior; conversion patterns may shift over time (e.g., with new marketing channels or program offerings), so the model should be periodically retrained/monitored rather than treated as permanent.

## 29. Future Improvements

- Retrain and monitor the model periodically as new lead data accumulates, to catch any drift in conversion patterns.
- Explore probability-threshold tuning (rather than the default 0.5 cutoff) to better balance Recall and Precision for the sales team's actual capacity.
- Investigate the `referral` channel further with a larger sample, given its promising but currently small-sample signal.
- Add authentication/rate-limiting to the Flask API before any real production deployment.
- Consider a lightweight model-monitoring/logging layer to track prediction distributions and flag drift over time.
- Evaluate additional algorithms (e.g., gradient boosting variants) if further performance gains are needed beyond Random Forest.

---

## GitHub & Large File Note

The serialized model (`extraalearn_lead_conversion_model.joblib`) is approximately **6.1 MB**, well under GitHub's default 100 MB per-file hard limit and its 50 MB "large file" warning threshold, so it can be committed to a standard GitHub repository without needing Git LFS. It has **not** been removed or excluded from this project — both the `models/` and `backend/models/` copies are included and confirmed byte-for-byte identical (see the Git Preparation section in the accompanying instructions/notebook).

---

## Acknowledgment

Dataset and business context provided as part of an academic ML deployment project based on ExtraaLearn's lead-conversion use case.
