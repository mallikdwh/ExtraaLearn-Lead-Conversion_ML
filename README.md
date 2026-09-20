<div align="center">

# 🚀 ExtraaLearn Lead Conversion Prediction

### Machine Learning • Flask REST API • Streamlit • Docker • Cloud Deployment

Predicting high-potential leads and transforming machine learning insights into an actionable sales-prioritization tool.

<br>

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask\&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63.0-red?logo=streamlit\&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker\&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/Project-Academic%20%2F%20Portfolio-lightgrey)](#)

<br>

### 🌐 Live Application

**Streamlit Frontend:**
https://extraalearn-lead-conversionml-tfvx8gbwa8dbnj8n5mvmd4.streamlit.app/

**Render Backend API:**
https://extraalearn-lead-conversion-api.onrender.com

</div>

---

## 📌 Table of Contents

* [🎯 Business Problem](#-business-problem)
* [💡 Project Objective](#-project-objective)
* [📂 Dataset](#-dataset)
* [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
* [📊 Key EDA Findings](#-key-eda-findings)
* [🛠️ Data Preprocessing](#️-data-preprocessing)
* [🚨 Outlier Treatment](#-outlier-treatment)
* [🤖 Modeling Approach](#-modeling-approach)
* [📈 Model Evaluation](#-model-evaluation)
* [🏆 Final Model](#-final-model)
* [🔎 Feature Importance](#-feature-importance)
* [💼 Business Insights](#-business-insights)
* [🚀 Deployment Architecture](#-deployment-architecture)
* [🌐 Live Deployment](#-live-deployment)
* [🔌 API Documentation](#-api-documentation)
* [🐳 Docker](#-docker)
* [🧪 End-to-End Validation](#-end-to-end-validation)
* [📁 Project Structure](#-project-structure)
* [⚙️ Environment & Reproducibility](#️-environment--reproducibility)
* [⚠️ Limitations](#️-limitations)
* [🔮 Future Improvements](#-future-improvements)
* [📎 GitHub & Model File](#-github--model-file)
* [🙏 Acknowledgment](#-acknowledgment)

---

## 🎯 Business Problem

ExtraaLearn is an early-stage EdTech startup offering programs on cutting-edge technologies to help students and professionals upskill and reskill.

The company generates leads through multiple channels, including:

* 🌐 Website
* 📱 Mobile App
* 📣 Digital Media
* 📰 Print Media
* 🎓 Educational Channels
* 🤝 Referrals

However, not every lead converts into a paying customer.

The sales team has limited time and resources, creating a need to identify **which leads are more likely to convert** so that follow-up efforts can be prioritized effectively.

---

## 💡 Project Objective

The objective of this project is to build and deploy a machine learning solution that can:

1. Predict whether a lead is likely to convert.
2. Estimate the probability of conversion.
3. Identify important conversion-related signals.
4. Develop a practical lead-prioritization workflow.
5. Expose the trained model through a REST API.
6. Provide a user-friendly Streamlit interface.
7. Deploy the complete solution to the cloud.

---

# 📂 Dataset

### Dataset Overview

| Attribute      | Details           |
| -------------- | ----------------- |
| Dataset        | `ExtraaLearn.csv` |
| Rows           | **4,612**         |
| Columns        | **15**            |
| Missing Values | **None**          |
| Duplicate Rows | **None**          |
| Unique IDs     | **4,612**         |
| Target         | `status`          |

### Target Distribution

| Status | Meaning       | Count | Percentage |
| -----: | ------------- | ----: | ---------: |
|    `0` | Not Converted | 3,235 |     70.14% |
|    `1` | Converted     | 1,377 |     29.86% |

### Overall Conversion Rate

> **29.86%**

The target is moderately imbalanced, which influenced the evaluation strategy.

---

# 🔍 Exploratory Data Analysis

The exploratory analysis included both univariate and bivariate analysis.

### Numerical Features

* `age`
* `website_visits`
* `time_spent_on_website`
* `page_views_per_visit`

### Categorical Features

* `current_occupation`
* `first_interaction`
* `profile_completed`
* `last_activity`
* `print_media_type1`
* `print_media_type2`
* `digital_media`
* `educational_channels`
* `referral`

### EDA Techniques

* Distribution plots
* Boxplots
* Count plots
* Conversion-rate tables
* Stacked bar charts
* Numerical feature comparison by conversion status

---

# 📊 Key EDA Findings

### ⏱️ Website Engagement

`time_spent_on_website` was the clearest behavioral difference.

| Status        | Median Time | Mean Time |
| ------------- | ----------: | --------: |
| Not Converted |     317 sec |   577 sec |
| Converted     |     789 sec | 1,068 sec |

Converted leads spent substantially more time on the website.

---

### 🌐 First Interaction

| First Interaction | Conversion Rate |
| ----------------- | --------------: |
| Website           |      **45.59%** |
| Mobile App        |      **10.53%** |

Website-origin leads showed a substantially higher observed conversion rate.

---

### ✅ Profile Completion

| Profile Completion | Conversion Rate |
| ------------------ | --------------: |
| High               |      **41.78%** |
| Medium             |      **18.88%** |
| Low                |       **7.48%** |

Higher profile completion was strongly associated with conversion.

---

### 👔 Current Occupation

| Occupation   | Conversion Rate |
| ------------ | --------------: |
| Professional |      **35.51%** |
| Unemployed   |      **26.58%** |
| Student      |      **11.71%** |

---

### 📱 Last Activity

| Last Activity    | Conversion Rate |
| ---------------- | --------------: |
| Website Activity |      **38.45%** |
| Email Activity   |      **30.33%** |
| Phone Activity   |      **21.31%** |

---

### 🤝 Referral

| Referral | Conversion Rate |
| -------- | --------------: |
| Yes      |      **67.74%** |
| No       |      **29.08%** |

Referral leads showed a particularly high observed conversion rate, although referrals represented only **93 leads**, so this signal should be interpreted cautiously.

> **Important:** EDA findings represent associations observed in the dataset and should not be interpreted as causal relationships.

---

# 🛠️ Data Preprocessing

### Feature Selection

The following were excluded:

* `ID` — unique row identifier
* `status` — target variable

This left **13 predictor variables**.

### Train/Test Split

A stratified 80/20 split was used:

```text
test_size = 0.20
random_state = 42
stratify = y
```

| Dataset  |  Rows |
| -------- | ----: |
| Training | 3,689 |
| Testing  |   923 |

Stratification preserved the approximately 70/30 target distribution in both datasets.

---

## 🔢 Numerical Pipeline

Features:

```text
age
website_visits
time_spent_on_website
page_views_per_visit
```

Processing:

```text
SimpleImputer(strategy="median")
            ↓
StandardScaler()
```

---

## 🔤 Categorical Pipeline

Features:

```text
current_occupation
first_interaction
profile_completed
last_activity
print_media_type1
print_media_type2
digital_media
educational_channels
referral
```

Processing:

```text
SimpleImputer(strategy="most_frequent")
            ↓
OneHotEncoder(handle_unknown="ignore")
```

All preprocessing was placed inside the model pipeline to prevent data leakage.

---

# 🚨 Outlier Treatment

IQR-based analysis identified potential outliers in:

* `website_visits`
* `page_views_per_visit`

However, these observations represented plausible high-engagement behavior rather than obvious data errors.

Therefore:

> **Outliers were retained.**

Tree-based ensemble models are relatively robust to extreme values, and removing legitimate high-engagement leads could have removed useful predictive information.

---

# 🤖 Modeling Approach

Two ensemble classification algorithms were evaluated:

### 🌲 Random Forest

`RandomForestClassifier`

### 🚀 AdaBoost

`AdaBoostClassifier`

Both models were implemented inside complete preprocessing pipelines.

---

# 📈 Model Evaluation

## Primary Metric — Recall

The primary evaluation metric was **Recall for the Converted class**.

The reason is business-oriented:

* **False Negative:** An actual potential customer is missed.
* **False Positive:** A non-converter receives additional sales attention.

For this use case, missing a genuine converter can represent a larger opportunity cost than following up with some non-converters.

Secondary metrics included:

* Accuracy
* Precision
* F1 Score
* ROC-AUC

---

## Baseline Results

| Model         | CV Recall | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------- | --------: | -------: | --------: | -----: | -----: | ------: |
| Random Forest |    0.7184 |   0.8570 |    0.7857 | 0.7174 | 0.7500 |  0.9166 |
| AdaBoost      |    0.6322 |   0.8321 |    0.7318 | 0.6920 | 0.7114 |  0.9005 |

Random Forest demonstrated stronger baseline performance across the reported metrics.

---

## Hyperparameter Tuning

### Random Forest

Tuned using:

* `RandomizedSearchCV`
* 25 sampled combinations
* 5-fold `StratifiedKFold`
* `scoring="recall"`
* `random_state=42`

Parameters included:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

### AdaBoost

Tuned using:

* `GridSearchCV`
* 12 parameter combinations
* 5-fold stratified cross-validation

The test set remained untouched during hyperparameter search.

---

## Tuned Results

| Model                   |  CV Recall |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| **Tuned Random Forest** | **0.7311** | **0.8613** | **0.7960** | **0.7210** | **0.7567** | **0.9232** |
| Tuned AdaBoost          |     0.6422 |     0.8321 |     0.7318 |     0.6920 |     0.7114 |     0.9008 |

---

# 🏆 Final Model

The final model is a **Tuned Random Forest classifier**.

### Final Configuration

```text
n_estimators = 100
max_depth = None
min_samples_split = 5
min_samples_leaf = 1
max_features = "sqrt"
bootstrap = True
criterion = "gini"
random_state = 42
```

The complete preprocessing and model pipeline was serialized using `joblib`.

---

## 📊 Final Test Performance

The final model was evaluated on the untouched test set of **923 leads**.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **0.8613** |
| Precision | **0.7960** |
| Recall    | **0.7210** |
| F1 Score  | **0.7567** |
| ROC-AUC   | **0.9232** |

---

# 🔎 Feature Importance

Feature importance was extracted from the fitted Random Forest after correctly mapping the one-hot encoded features back to their transformed names.

| Rank | Feature                           | Importance |
| ---: | --------------------------------- | ---------: |
| 🥇 1 | `time_spent_on_website`           | **0.2516** |
| 🥈 2 | `first_interaction_Website`       | **0.1178** |
| 🥉 3 | `age`                             | **0.0904** |
|    4 | `page_views_per_visit`            | **0.0901** |
|    5 | `first_interaction_Mobile App`    | **0.0853** |
|    6 | `profile_completed_High`          | **0.0701** |
|    7 | `website_visits`                  | **0.0536** |
|    8 | `profile_completed_Medium`        | **0.0477** |
|    9 | `last_activity_Phone Activity`    | **0.0319** |
|   10 | `current_occupation_Professional` | **0.0303** |

> Feature importance indicates predictive usefulness within the model. It does not establish causation.

---

# 💼 Business Insights

The analysis suggests several practical ways ExtraaLearn could prioritize sales activity.

### 1. ⏱️ Prioritize Strong Website Engagement

Time spent on the website was the strongest model feature.

Leads demonstrating stronger engagement may warrant earlier sales follow-up.

### 2. ✅ Encourage Profile Completion

Profile completion showed a strong observed relationship with conversion.

Improving the completion rate could potentially provide additional useful lead information.

### 3. 🌐 Monitor First-Interaction Channels

Website-origin leads had a substantially higher observed conversion rate than Mobile App leads.

Channel-level performance should therefore be monitored when allocating marketing and sales resources.

### 4. 📱 Incorporate Recent Activity

Recent lead activity can provide useful prioritization signals.

### 5. 🤝 Investigate Referral Leads

Referral leads showed a high observed conversion rate, but the sample size was small.

Additional data should be collected before making larger strategic decisions based on this signal.

### 6. 🎯 Use Predictions for Prioritization

The model should be used to **prioritize outreach**, not automatically reject lower-probability leads.

---

# 🚀 Deployment Architecture

The project uses a decoupled deployment architecture.

```text
                    ┌───────────────────────┐
                    │       End User        │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Streamlit Frontend    │
                    │ Community Cloud       │
                    └───────────┬───────────┘
                                │
                         HTTPS POST /predict
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Flask REST API        │
                    │ Render                │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Serialized Pipeline  │
                    │ scikit-learn 1.6.1   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Tuned Random Forest  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Prediction +          │
                    │ Conversion Probability│
                    └───────────────────────┘
```

---

# 🌐 Live Deployment

## Frontend — Streamlit Community Cloud

### 🔗 Live Application

**ExtraaLearn Lead Conversion Prediction**

https://extraalearn-lead-conversionml-tfvx8gbwa8dbnj8n5mvmd4.streamlit.app/

The frontend:

* Collects the 13 lead attributes.
* Sends the input to the Flask API.
* Displays the prediction.
* Displays the conversion probability.
* Handles API errors and connection failures.

---

## Backend — Render

### 🔗 Live API

**ExtraaLearn Lead Conversion Prediction API**

https://extraalearn-lead-conversion-api.onrender.com

### Health Check

```text
GET /
```

Expected response:

```json
{
  "message": "ExtraaLearn Lead Conversion Prediction API is running",
  "status": "healthy"
}
```

The deployed API was successfully tested after deployment.

---

# 🔌 API Documentation

## `GET /`

Health check endpoint.

### Response

```json
{
  "message": "ExtraaLearn Lead Conversion Prediction API is running",
  "status": "healthy"
}
```

---

## `POST /predict`

Accepts 13 lead attributes.

| Field                   | Type    | Example            |
| ----------------------- | ------- | ------------------ |
| `age`                   | Integer | `35`               |
| `current_occupation`    | String  | `"Professional"`   |
| `first_interaction`     | String  | `"Website"`        |
| `profile_completed`     | String  | `"High"`           |
| `website_visits`        | Integer | `10`               |
| `time_spent_on_website` | Integer | `1000`             |
| `page_views_per_visit`  | Float   | `6.0`              |
| `last_activity`         | String  | `"Phone Activity"` |
| `print_media_type1`     | String  | `"Yes"`            |
| `print_media_type2`     | String  | `"Yes"`            |
| `digital_media`         | String  | `"Yes"`            |
| `educational_channels`  | String  | `"Yes"`            |
| `referral`              | String  | `"Yes"`            |

---

# 🧪 Example API Request

```json
{
  "age": 35,
  "current_occupation": "Professional",
  "first_interaction": "Website",
  "profile_completed": "High",
  "website_visits": 10,
  "time_spent_on_website": 1000,
  "page_views_per_visit": 6.0,
  "last_activity": "Phone Activity",
  "print_media_type1": "Yes",
  "print_media_type2": "Yes",
  "digital_media": "Yes",
  "educational_channels": "Yes",
  "referral": "Yes"
}
```

---

# 📤 Example API Response

The example request was tested against the live Render API.

```json
{
  "conversion_probability": 0.7223,
  "prediction": 1,
  "prediction_label": "Converted"
}
```

### Result

> **Conversion Probability: 72.23%**
> **Prediction: Converted**

The same test case produced approximately **72.2%** through the deployed Streamlit application.

---

# 🐳 Docker

The backend is containerized using Docker.

### Backend Docker Stack

```text
Python 3.11
     ↓
Flask
     ↓
Gunicorn
     ↓
scikit-learn 1.6.1
     ↓
Serialized Random Forest Pipeline
```

### Build Backend

```bash
docker build -t extraalearn-backend ./backend
```

### Run Backend

```bash
docker run -p 5000:5000 extraalearn-backend
```

The backend Docker configuration was successfully deployed through Render.

### Frontend Docker

A Dockerfile is also included for the Streamlit frontend.

```bash
docker build -t extraalearn-frontend ./frontend
```

```bash
docker run -p 8501:8501 \
  -e BACKEND_URL=http://host.docker.internal:5000/predict \
  extraalearn-frontend
```

The production frontend deployment uses **Streamlit Community Cloud** rather than the frontend Docker container.

---

# 🧪 End-to-End Validation

The complete deployed workflow was tested successfully.

### ✅ Backend Health

```text
Status: healthy
```

### ✅ Direct API Prediction

```text
Prediction: Converted
Conversion Probability: 72.23%
```

### ✅ Streamlit Prediction

```text
Prediction: Converted
Conversion Probability: 72.2%
```

### ✅ Model Compatibility

Final serialized model validated with:

```text
scikit-learn: 1.6.1
joblib: 1.6.0
```

### ✅ Model Integrity

The model stored in:

```text
models/
backend/models/
```

was verified to be byte-for-byte identical.

---

# 📁 Project Structure

```text
ExtraaLearn-Lead-Conversion_ML/
│
├── 📓 notebook/
│   └── ExtraaLearn_Model_Deployment_Additional_Project_Full_Code.ipynb
│
├── 📊 data/
│   └── ExtraaLearn.csv
│
├── 🤖 models/
│   └── extraalearn_lead_conversion_model.joblib
│
├── 🔌 backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   └── models/
│       └── extraalearn_lead_conversion_model.joblib
│
├── 🖥️ frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── 📄 README.md
│
└── 🚫 .gitignore
```

---

# ⚙️ Environment & Reproducibility

## Final Model Environment

```text
Python: 3.11+
scikit-learn: 1.6.1
joblib: 1.6.0
```

The final model was regenerated and validated using **scikit-learn 1.6.1**.

This version is also pinned in the backend deployment requirements to ensure model compatibility.

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### Local Architecture

```text
Browser
   ↓
Streamlit :8501
   ↓
Flask :5000
   ↓
ML Pipeline
   ↓
Prediction
```

---

# 🔐 Environment Variables

The frontend uses:

```text
BACKEND_URL
```

### Local Default

```text
http://localhost:5000/predict
```

### Cloud Deployment

The production backend URL is configured through **Streamlit Community Cloud Secrets** rather than being hardcoded into the source code.

```toml
BACKEND_URL = "https://extraalearn-lead-conversion-api.onrender.com/predict"
```

No credentials, passwords, API keys, or sensitive configuration files are committed to GitHub.

---

# ⚠️ Limitations

* The final model Recall is **0.7210**, meaning some actual converters are still missed.
* The model is intended as a lead-prioritization tool rather than a perfect classification system.
* Referral conversion rate is based on a relatively small sample.
* Feature importance represents predictive association, not causality.
* Historical conversion patterns may change as marketing channels and customer behavior evolve.
* The deployed API does not currently include authentication or rate limiting and should be considered a portfolio/academic deployment rather than an enterprise production service.
* The free Render service may take longer to respond after inactivity. During validation, the first request timed out, while a subsequent request completed successfully.

---

# 🔮 Future Improvements

Potential future improvements include:

* 🔄 Periodic model retraining
* 📊 Model and data drift monitoring
* 🎚️ Probability-threshold optimization
* 🔬 Additional feature engineering
* 🤖 Evaluation of additional boosting algorithms
* 🔐 API authentication
* 🚦 API rate limiting
* 📋 Structured API logging
* 🧪 Automated unit and integration testing
* 🔄 CI/CD pipeline
* 📈 Production monitoring dashboard
* 🗃️ Larger and more recent lead datasets

---

# 📎 GitHub & Model File

The serialized model is approximately **6.1 MB**, which is well below GitHub's 100 MB per-file limit.

The trained model is included in the repository:

```text
models/extraalearn_lead_conversion_model.joblib
```

and is copied to the backend deployment directory:

```text
backend/models/extraalearn_lead_conversion_model.joblib
```

Both files were verified to be identical.

### Model Compatibility

```text
scikit-learn 1.6.1
joblib 1.6.0
```

---

# 🙏 Acknowledgment

Dataset and business context were provided as part of an academic machine-learning deployment project based on the ExtraaLearn lead-conversion use case.

---

<div align="center">

### ⭐ If you found this project useful, consider giving the repository a star!

**Machine Learning → API → UI → Cloud Deployment**

</div>
