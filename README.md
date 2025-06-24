# -Customer-Churn-Analysis-for-Telecom-Industry

Project Summary
This repository presents a machine learning solution for predicting customer churn in the telecom industry. Using real-world data, we conducted end-to-end processing — from exploratory data analysis to model building, evaluation, and interpretability. Our goal: enable proactive retention strategies by accurately identifying customers at risk of leaving.

Tools & Technologies
Programming: Python (Google colab)
Libraries: Pandas, NumPy, Scikit-learn, ELI5
Data Visualization: Matplotlib, Seaborn
Dataset: telecom_customer_churn_data.csv
Other: Excel (for initial dataset handling)
Mysql for data aggregation
Machine learning for prediction and better accuracy.

Key Project Features
Comprehensive EDA to understand customer behavior patterns
Preprocessing pipelines: handling missing values, feature encoding, scaling
Model Development: Logistic Regression, Random Forest Classifier
Model Explainability: SHAP analysis to interpret key churn drivers
Customer Segmentation: Classifying customers into At Risk, Loyal, and Dormant
Actionable Insights: Business recommendations to reduce churn rates


Future Work
Hyperparameter tuning with GridSearchCV
Deep Learning modeling (ANNs)
Deployment via Flask API / Streamlit dashboard
Further analysis using SHAP for richer interpretability

Customer Churn Project – 20 Key Points
The project focuses on predicting customer churn in the telecom industry.

Churn refers to customers discontinuing their telecom service.

SQL was used to aggregate features like call duration, recharge count, and complaints.

Python was used for preprocessing, visualization, and modeling.

The dataset included customer usage, complaints, and churn status.

Missing values and duplicates were cleaned from the data.

Categorical features were one-hot encoded.

Numerical features were scaled using StandardScaler.

EDA showed that churners had low usage and high complaints.

Customers with fewer recharges were more likely to churn.

Feature engineering created new variables like average recharge and complaint rate.

PCA was used to reduce dimensionality and visualize the data.

PCA reduced 30+ features to 3 while preserving 90% of data variance.

Machine learning models trained: Logistic Regression, Ridge, Lasso, ElasticNet.

ElasticNet model performed the best with an R² score of 0.82.

GridSearchCV was used to optimize hyperparameters.

SHAP was used to interpret the model and identify key churn drivers.

Complaint count, recharge frequency, and call duration were top features.

Customers were segmented into At Risk (18%), Loyal (62%), and Dormant (20%).

Final recommendations include early engagement, better service, and personalized offers.

