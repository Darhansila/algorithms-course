# Machine Learning Algorithms & Applied Data Analysis

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Comprehensive collection of practical laboratory projects and implementations of foundational Machine Learning algorithms developed as part of the **Advanced Algorithms (Course I)** curriculum at Satbayev University.

Each laboratory includes clean datasets, Jupyter Notebook implementations (`.ipynb`), high-resolution visualization charts, and full formal academic reports (`.docx`).

---

## 📑 Projects Summary

| Module | Algorithm | Problem Domain | Target & Task | Key Metrics |
|:---:|---|---|---|:---:|
| **[Practice 1](./1practice/)** | **Linear Regression (OLS)** | Employee Compensation | Predict salary based on age and tenure | $R^2 = 0.89$, $\text{MSE}$ |
| **[Practice 2](./2practice/)** | **Gaussian Naive Bayes** | E-commerce Customer Behavior | Predict purchase conversion from session activity | $\text{Accuracy} = 76.0\%$, $\text{AUC}$ |
| **[Practice 3](./3practice/)** | **Logistic Regression** | Cybersecurity / Email Filtering | Binary spam classification via structural email features | $\text{Accuracy} = 96.7\%$, $\text{F1} = 0.966$ |
| **[Practice 4](./4practice/)** | **K-Nearest Neighbors (KNN)** | Email Spam Classification | Non-parametric classification with feature scaling & $k$-tuning | $\text{Accuracy} = 94.0\%$, $\text{Recall} = 96.0\%$ |
| **[Practice 5](./5practice/)** | **Support Vector Machines (SVM)** | Academic Analytics | Predict student exam success from attendance and study metrics | $\text{Accuracy} = 94.0\%$--$98.0\%$ across kernels |

---

## 🔬 Detailed Practice Descriptions

### 1. Practice 1 — Linear Regression (`1practice/`)
* **Objective:** Study ordinary least squares (OLS) linear regression for univariate continuous target estimation.
* **Dataset:** [`sotrud.csv`](./1practice/sotrud.csv) — Employee demographics, age, and compensation figures.
* **Methodology:** 
  * Model fitting via `sklearn.linear_model.LinearRegression`.
  * Regression slope ($\beta_1$) and intercept ($\beta_0$) extraction and interpretation.
  * Model quality evaluation using the coefficient of determination ($R^2$).
  * Matplotlib visualization of the linear regression trend against ground-truth scatter points.
* **Artifacts:**
  * Notebook: [`pr/Untitled14.ipynb`](./1practice/pr/Untitled14.ipynb)
  * Report: [`pr/алгоритмы 1 практика отчет.docx`](./1practice/pr/алгоритмы%201%20практика%20отчет.docx)

---

### 2. Practice 2 — Naive Bayes Classification (`2practice/`)
* **Objective:** Probabilistic modeling using Bayes' Theorem under conditional feature independence assumptions.
* **Dataset:** [`data.csv`](./2practice/data.csv) (250 rows) — Real-world structured customer purchase dataset including demographics (`full_name`, `gender`, `age`, `city`), historical metrics (`total_orders`, `avg_order_value`), and session behavioral signals (`product_views`, `cart_adds`, `visits_last_month`).
* **Methodology:**
  * Implementation of `sklearn.naive_bayes.GaussianNB`.
  * Multi-split stability evaluation (80/20, 70/30, 90/10).
  * Feature importance analysis revealing `cart_adds` and `visits_last_month` as primary predictors.
  * Confusion Matrix heatmaps and posterior class probability calculation (`predict_proba`).
* **Artifacts:**
  * Notebook: [`2practice.ipynb`](./2practice/2practice.ipynb)
  * Visualizations: [`generated_images/`](./2practice/generated_images/)
  * Report: [`Практика2Алгоритм.docx`](./2practice/Практика2Алгоритм.docx)

---

### 3. Practice 3 — Logistic Regression (`3practice/`)
* **Objective:** Binary classification using logistic sigmoid function with L2 regularization and feature standardization.
* **Dataset:** [`data.csv`](./3practice/data.csv) (300 rows) — Structural email metadata including text length, link frequency, exclamation count, uppercase word count, spam trigger words, currency symbols, and sender contact status.
* **Methodology:**
  * Feature scaling using `StandardScaler` to balance coefficient penalties.
  * Model training via `sklearn.linear_model.LogisticRegression(max_iter=1000)`.
  * Extraction and interpretation of feature log-odds weights (`coef_`): positive drivers (`spam_word_count`, `has_currency`, `num_links`) vs. negative indicators (`sender_in_contacts`, `is_reply`).
  * Confusion matrix heatmap and classification report evaluation.
* **Artifacts:**
  * Notebook: [`3practice.ipynb`](./3practice/3practice.ipynb)
  * Visualizations: [`report_images/`](./3practice/report_images/)
  * Report: [`Практика3Алгоритм.docx`](./3practice/Практика3Алгоритм.docx)

---

### 4. Practice 4 — K-Nearest Neighbors Classifier (`4practice/`)
* **Objective:** Instance-based learning and non-parametric classification using Minkowski/Euclidean metric spaces.
* **Dataset:** [`data.csv`](./4practice/data.csv) (250 rows) — Complete, realistic email communications dataset with sender identity attributes (`full_name`, `sender_email`, `gender`, `birth_year`, `city`, `subject`, `date_sent`) paired with 10 numeric classification features.
* **Methodology:**
  * Stratified 80/20 train/test partitioning and `StandardScaler` normalization.
  * Hyperparameter grid exploration over $k \in [1..20]$ to balance bias-variance tradeoff.
  * Optimal $k=5$ selection yielding:
    * **Accuracy:** $94.00\%$
    * **Precision:** $92.31\%$
    * **Recall:** $96.00\%$
    * **F1-Score:** $94.12\%$
  * Detailed Confusion Matrix diagnosis ($TN=23, FP=2, FN=1, TP=24$) and probabilistic inference on unseen emails.
* **Artifacts:**
  * Notebook: [`4practice.ipynb`](./4practice/4practice.ipynb)
  * Visualizations: [`report_images/`](./4practice/report_images/)
  * Report: [`Практика4Алгоритм.docx`](./4practice/Практика4Алгоритм.docx)

---

### 5. Practice 5 — Support Vector Machines (`5practice/`)
* **Objective:** Maximum margin hyperplane separation, kernel transformations, and soft-margin regularization.
* **Dataset:** [`data.csv`](./5practice/data.csv) (250 rows) — Academic engagement dataset with student demographics (`student_id`, `full_name`, `gender`, `birth_year`, `group`, `city`), core academic metrics (`attendance` %, `study_hours`, `test_score`), and final examination pass outcome (`result`).
* **Methodology:**
  * Comparative evaluation across three kernel architectures:
    * **Linear:** $\text{Accuracy} = 94.00\%$, $\text{Precision} = 95.83\%$, Support Vectors: 27
    * **RBF:** $\text{Accuracy} = 94.00\%$, $\text{Precision} = 95.83\%$, Support Vectors: 39
    * **Polynomial ($d=3$):** $\text{Accuracy} = 94.00\%$, $\text{Precision} = 100.0\%$, Support Vectors: 56
  * Regularization parameter analysis ($C \in [0.1, 1.0, 10.0, 100.0]$) demonstrating margin narrowing and support vector reduction from 81 down to 26 vectors.
  * Dual-sample probability estimation (`predict_proba` via Platt scaling) predicting student pass/fail with $>99.7\%$ certainty.
* **Artifacts:**
  * Notebook: [`5practice.ipynb`](./5practice/5practice.ipynb)
  * Visualizations: [`report_images/`](./5practice/report_images/)
  * Report: [`Практика5Алгоритм.docx`](./5practice/Практика5Алгоритм.docx)

---

## 🛠️ Environment Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Darhansila/algorithms-course.git
cd algorithms-course
```

### 2. Set up virtual environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl python-docx jupyter
```

### 4. Run Jupyter Lab / Notebooks
```bash
jupyter notebook
```
Open any of the practice directories (`1practice`, `2practice`, `3practice`, `4practice`, `5practice`) and run the respective `.ipynb` notebooks.

---

## 👤 Author

**[Имя удалено по соображениям конфиденциальности]** ([@Darhansila](https://github.com/Darhansila))  
Computer Science | Satbayev University
