# **Housing Prices Prediction**

## **🌟 Overview**
Housing Prices Prediction is a machine learning project aimed at predicting housing prices in California. It involves comparing various machine learning models (Linear Regression, Decision Tree, and Random Forest) to determine the most effective algorithm for estimating house prices. The project explores various data analysis techniques, such as feature scaling, data preprocessing, and model evaluation using metrics like RMSE (Root Mean Squared Error) to assess model performance.

## **❓ Why This Project?**
The primary goal of this project is to apply machine learning techniques to predict housing prices using various attributes such as median income, population density, and location. By comparing different models, the project aims to identify which algorithm yields the most accurate results, helping to understand how well each model can predict housing prices and contributing to the field of real estate analytics.

## **🔑 Key Features & Learnings**
* Multiple Model Comparison: The project compares three models: Linear Regression, Decision Tree, and Random Forest.
* Data Preprocessing: Handling missing values, transforming categorical attributes, and scaling numerical features for optimal model performance.
* Overfitting & Underfitting: The project discusses these concepts and how they affect model accuracy, using techniques like K-fold cross-validation to mitigate them.
* Model Evaluation: RMSE (Root Mean Squared Error) and cross-validation are used to measure model performance.

## **🛠 Technologies, Tools, and Frameworks**
* Machine Learning Algorithms: Linear Regression, Decision Tree, Random Forest
* Data Analysis: Python, Pandas, Matplotlib, Seaborn
* Data Preprocessing: scikit-learn (Pipeline, Imputer, FeatureUnion)
* Model Evaluation: RMSE, K-fold Cross-Validation, Grid Search
* Data Storage: CSV files (for the dataset)
* Python Libraries: pandas, scikit-learn, numpy, matplotlib, seaborn

## **🚀 Data Source**
The dataset used is the "California Housing Prices" dataset from the Statlib repository, based on 1990 U.S. Census data. The dataset contains features like population, median income, and median dwelling price, which are used to predict house prices in California.

## **👉 Installation & Usage**
Install Python Libraries--> Download Dataset--> Run the Code: Once the dataset is downloaded and dependencies are installed, run the Python script to train the models and evaluate the results.

**Quick Start**
1. Load the dataset
2. Preprocess the data:
  *Handle missing values
  *Apply feature scaling using Standardization or Min-Max Scaling
3. Train and Evaluate Models:
  *Use Linear Regression, Decision Tree, and Random Forest models.
  *Evaluate performance using RMSE and K-fold cross-validation.
4. Model Tuning:
  *Apply Grid Search to find optimal hyperparameters for Random Forest.

## **📊 Exploratory Data Analysis (EDA)**
* Visualizing Distributions: Plot histograms for numerical features such as median income and house prices.
* Geographical Analysis: Use scatter plots to examine correlations between latitude, longitude, and housing prices.
* Correlation Analysis: Investigate correlations between features like median income and house price values using heatmaps and scatter plots.

## **🔍 Key Visualizations**
1. Scatterplot of House Prices vs. Location: Visualizes how house prices correlate with geographical features.
2. Histograms: Shows distributions of features like median income, median house value, and total rooms.
3. Correlation Heatmap: Displays the strength of relationships between various features and the target variable (house prices).
4. Model Performance Graphs: Comparing model performance using RMSE on training and validation sets.

## **🔬 Model Selection & Evaluation**
1. Linear Regression: Gives a baseline for prediction but underperforms with RMSE = 68376.5125.
2. Decision Tree: Overfits the data with RMSE = 0, which was mitigated using cross-validation.
3. Random Forest: Produces the best results with a RMSE score of 50,804, improved through hyperparameter tuning via Grid Search.

## **🛠 Feature Engineering & Selection**
1. Feature Scaling: Standardized numerical attributes to avoid performance degradation due to widely varying scales.
2. Handling Categorical Data: Transformed the ocean proximity feature into numerical values.
3. Imputation: Used the median value to fill missing data for attributes like total_bedrooms.

## **⚡ Hyperparameter Tuning**
Grid Search: Used to fine-tune hyperparameters such as max_features and n_estimators for the Random Forest Regressor, which led to an improved RMSE of 48,557.

## **🎯 Key Insights & Conclusions**
After hyperparameter tuning, the Random Forest model provided the most accurate predictions for housing prices. Feature correlation analysis revealed that median income had the strongest relationship with house prices, highlighting its importance as a predictor. During the model evaluation, it was observed that Linear Regression suffered from underfitting, failing to capture the complexity of the data, while the Decision Tree model exhibited overfitting, performing well on training data but poorly on unseen data. These issues were addressed by utilizing more complex models, such as Random Forest, and employing cross-validation techniques to improve model generalization and accuracy.

## **🔮 Future Improvements**
* Advanced Models: Implement more complex models like Gradient Boosting or XGBoost for potentially better performance.
* Feature Engineering: Incorporate additional features such as crime rates, school ratings, and transportation availability.
* Outlier Removal: Investigate and remove extreme outliers that may skew model performance.
