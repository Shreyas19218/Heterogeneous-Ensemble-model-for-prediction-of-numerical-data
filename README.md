# 🎓 Heterogeneous Ensemble Model for Student Exam Performance Forecasting

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-blue.svg)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-purple.svg)

## 📌 Project Overview

This project focuses on predicting continuous student exam scores using pre-exam behavioral, demographic, and academic indicators.

The project benchmarks traditional machine learning algorithms, including Decision Trees and Support Vector Regression (SVR), against advanced gradient boosting models such as XGBoost and LightGBM. It also evaluates heterogeneous ensemble approaches, including Voting and Stacking Regressors, to compare their predictive performance.

The project implements an end-to-end machine learning pipeline covering data preprocessing, feature engineering, model training, performance evaluation, and model interpretability.

## 📊 Dataset & Exploratory Data Analysis

- **Dataset Source:** [Kaggle – Student Exam Performance Dataset](https://www.kaggle.com/datasets/ssssws/student-exam-performance-dataset)
- **Initial Dataset Size:** 100,000 records
- **Features:** 44 features
- **Sampled Dataset Size:** 10,000 records for experimentation
- **Target Variable:** `exam_score` (continuous numerical value)

### Data Visualizations

The following visualizations explore the distribution of exam scores, relationships between academic indicators, and correlations among numerical features.

**1. Distribution of Exam Scores**

![Distribution of Exam Scores](exam_score_dist.png)

*The distribution illustrates the frequency of student exam scores, with most observations concentrated around the middle of the score range.*

**2. Correlation Matrix of Key Numeric Features**

![Correlation Matrix](correlation_matrix.png)

*The correlation matrix highlights relationships between exam scores, previous academic performance, study hours, attendance, sleep hours, and previous GPA.*

**3. Previous Exam Score vs. Final Exam Score**

![Previous Exam Score vs Final Exam Score](prev_vs_final_score.png)

*The scatter plot shows a positive relationship between previous exam scores and final exam scores, suggesting that historical academic performance is an informative predictor.*

## 🛠️ Data Preprocessing & Feature Engineering

Data preprocessing was designed to improve model reliability and reduce the risk of data leakage.

1. **Target Leakage Removal:** Removed post-exam variables such as `questions_correct`, `questions_attempted`, `performance_grade`, and `pass_status` that could reveal information about the target.
2. **Missing Value Imputation:** Applied median imputation to numerical features such as `previous_gpa`, `attendance_percentage`, and `time_management_score`.
3. **Categorical Encoding:** Applied one-hot encoding using `pd.get_dummies()` with `drop_first=True` to transform categorical variables into numerical representations.
4. **Feature Scaling:** Applied `StandardScaler` to numerical features for distance-based and scale-sensitive models, particularly Support Vector Regression.
5. **Train-Test Split:** Evaluated models using an 80:20 training and testing split.

## 🧠 Model Architecture & Training

Multiple individual models and ensemble configurations were trained and evaluated.

### Individual Models

- **Decision Tree Regressor:** Evaluated tree depths of 7 and 8 to capture nonlinear relationships.
- **Support Vector Regressor (SVR):** Used an RBF kernel with `C=1.0` and `epsilon=0.1`.
- **XGBoost Regressor:** Applied gradient boosting to learn nonlinear feature interactions and correct prediction errors iteratively.
- **LightGBM Regressor:** Used a gradient boosting framework optimized for efficient training and predictive performance.

### Ensemble Models

- **Voting Regressor:** Combined SVR and Decision Tree predictions through averaging.
- **Stacking Regressor:** Combined two Decision Trees and one SVR, using Linear Regression as the meta-model with 5-fold cross-validation.
- **Mixed Ensemble:** Evaluated an additional combination of two Decision Trees and one SVR.

### Gradient Boosting Configuration

The XGBoost and LightGBM models were configured with the following parameters:

- `n_estimators = 100`
- `learning_rate = 0.1`
- `max_depth = 5`

## 📈 Model Performance Evaluation

Models were evaluated on a held-out test set using two regression metrics:

- **R² Score:** Measures the proportion of target variance explained by the model. Higher values generally indicate better predictive performance.
- **RMSE (Root Mean Squared Error):** Measures the typical magnitude of prediction errors, penalizing larger errors more heavily. Lower values indicate better performance.

| Model Configuration | R² Score | RMSE |
|:---|:---:|:---:|
| **LightGBM** | **0.6452** | **9.3866** |
| **XGBoost** | **0.6434** | **9.4098** |
| Stacking Ensemble (Linear Meta-Model) | 0.6039 | 9.9178 |
| Voting Ensemble (SVR + Decision Tree) | 0.5715 | 10.3158 |
| SVR (Individual) | 0.5713 | 10.3174 |
| Mixed Ensemble (2 Decision Trees + 1 SVR) | 0.5487 | 10.5866 |

### Key Findings

- **LightGBM achieved the best overall performance**, with an R² score of 0.6452 and an RMSE of 9.3866.
- XGBoost performed comparably, achieving an R² score of 0.6434.
- Both gradient boosting models outperformed the evaluated ensemble configurations.
- The results suggest that gradient boosting effectively captures nonlinear relationships in student academic and behavioral data.
- The remaining prediction error indicates that exam performance is influenced by factors that are not fully captured by the available features.

## 🔍 Model Interpretability & Feature Importance

To understand which features contribute most to exam score predictions, the project uses gain-based feature importance from LightGBM and XGBoost, along with SHAP (SHapley Additive exPlanations).

### 1. LightGBM Feature Importance by Gain

![Top 20 Feature Importances by Gain - LightGBM](Light_gbm_gain.jpeg)

The gain-based importance plot identifies `previous_exam_score` as the most influential feature, followed by `study_hours_per_day`, `exam_preparation_days`, and `exam_difficulty_Hard`.

### 2. XGBoost Feature Importance by Gain

![Top 20 Feature Importances by Gain - XGBoost](Xg_boost_gain.jpeg)

XGBoost also assigns high importance to `previous_exam_score`, `exam_difficulty_Hard`, `study_hours_per_day`, and `exam_difficulty_Medium`. This provides a complementary view of the features used to improve predictions.

### 3. SHAP Feature Importance Analysis

![SHAP Feature Importance Summary - XGBoost](Shap_analysis.jpeg)

The SHAP summary plot explains both the magnitude and direction of individual feature contributions to the model's predictions.

- **Horizontal position:** Represents the SHAP value. Positive values push predictions higher, while negative values push predictions lower.
- **Color:** Represents the feature value, with red indicating higher values and blue indicating lower values.
- **Distribution:** Shows how a feature's contribution varies across observations.

The SHAP analysis indicates that previous exam scores, study hours, practice tests completed, and exam preparation days have substantial influence on predictions. The effects of categorical features, such as exam difficulty, depend on their encoded values.

### Interpretability Insights

Across the importance plots, previous academic performance consistently emerges as a leading predictor. Study habits, preparation time, practice tests, attendance, and exam difficulty also contribute to model predictions.

These findings describe learned associations in the dataset; they do not establish that a feature directly causes an improvement in exam performance.

## 🧰 Technology Stack

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Gradient Boosting:** XGBoost, LightGBM
- **Model Interpretability:** SHAP

## 👨‍💻 Contributor

**Shreyas Snehal**

## ⭐ Conclusion

This project presents a comparative evaluation of traditional regression algorithms, gradient boosting models, and heterogeneous ensemble techniques for student exam score forecasting.

LightGBM achieved the strongest results among the evaluated models, with XGBoost delivering closely comparable performance. Feature importance and SHAP analysis further demonstrate how historical academic performance, study habits, and preparation-related indicators contribute to the models' predictions.

The project highlights the importance of leakage-aware preprocessing, systematic model evaluation, and interpretable machine learning when developing predictive systems using educational data.
