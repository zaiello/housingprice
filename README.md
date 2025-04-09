# 🏡 House Price Prediction with Regression Models

## 📌 Objective

Build a high-performance regression model to predict home sale prices using the **Ames Housing Dataset**. This project involed binarization, model tuning, and feature transformation in Python.

![Kaggle Competition]((https://www.kaggle.com/competitions/home-data-for-ml-course/data))


> 📈 RMSLE: **0.12984**  
> 🏆 **Ranked in the top 2% at time of submission**  
---

## 📚 Table of Contents
 
- [Datasets Used](#-datasets-used)  
- [Technologies Used](#-technologies-used)  
- [Step-by-Step Breakdown](#-step-by-step-breakdown)  
- [Results & Takeaways](#-results--takeaways)

---

## 📂 Datasets Used

The Ames Housing Dataset, compiled by Dean De Cock for use in data science education, describes the sale of individual residential property in Ames, Iowa from 2006 to 2010. The data set contains 2930 observations and a large number of explanatory variables (23 nominal, 23 ordinal, 14 discrete, and 20 continuous) involved in assessing home values.

This dataset was provided through the Kaggle competition split into training and testing sets:

- `train.csv`: Model training data: 81 fields - the unique ID field, 79 input fields, and one output field, SalePrice.
- `test.csv`: Unlabeled test data used for submission

---

## 🛠 Technologies Used

- **Python 3**
- **NumPy**, **Pandas** – Data processing
- **Scikit-learn** – Ridge Regression

---

## 🧠 Step-by-Step Breakdown

### Step 1: Feature Engineering

Machine learning models (especially linear ones like Linear Regression, k-NN, etc.) operate on numerical inputs. If we were to leave the categories as strings or even assign them numbers arbitrarily (e.g., Gable = 1, Hip = 2, Flat = 3), the model would mistakenly interpret a relationship or order between them — like "Hip" being greater than "Gable", which doesn’t make any logical sense. Naïve binarization (binarizing all features) assumes that all features are independent, which isn’t true in our dataset (ex. LotArea is positively correlated with SalePrice). Since the assumption of independence is not met, the accuracy of our predictions can be negatively affected. Naïve binarization also doesn’t handle NA (or missing) values differently from actual features which leads to data sparseness: using missing data to make predictions affects their accuracy. In addition to reducing the accuracy of our predictions, the use of missing data wastes space and computational power.

So we binarize:

✅ To make the data numeric so models can process it

✅ To prevent the model from making erroneous assumptions about relationships between categorical values

✅ To ensure each category has equal weight and independence in the model

✅ To ensure missing data are handled appropriately 

> LotFrontage and GarageYrBlt are mixed fields - they have both numeric and categorical elements
> These fields were treated as numeric to preserve their order and scale.

### Step 2: Missing Value Imputation

I used zero-imputation for missing values. I tried imputing missing values using mean and median but the RMSLE was higher than the zero-imputed model possibly due to feature distribution mismatches or losing regularization benefits of zero-imputation.  

### Step 3: Fit a Ridge Regression Model on Log-Transformed Outcome Variable

📌 Ridge Regression = Linear Regression + Regularization
At its core, Ridge Regression is just a linear regression model with a penalty added to prevent overfitting.

🎯 Why Use Ridge Regression?

✅ Helps with Overfitting
* In high-dimensional data (like our binarized features), the model might fit noise instead of signal.
* Ridge shrinks weights, which prevents wild predictions and improves generalization.
  
✅ Handles Multicollinearity
* If features are correlated (which is common with one-hot encodings), standard linear regression can have unstable coefficients.
* Ridge stabilizes this by shrinking correlated coefficients together.
  
✅ Improves Robustness in Sparse/Noisy Data
* Our dataset includes thousands of binary features and some NAs.
* Ridge provides a buffer that makes the model less sensitive to weird blips in the data.

🎯 Why Use The Log-Transformed Outcome?

The reason we tranform the outcome variable, SalePrice, is because we want to minimize RMSLE (Root Mean Squared Log Error), our evaluation metric. RMSLE is robust to outliers because it only considers the relative error between the expected and observed values deeming the magnitude of the errors insignificant. The RMLSE incurs a larger penalty for a prediction that is an underestimation.

For many contests, this last point is very important because overestimation can usually be tolerated. For example, let’s say that someone is using our model to appraise their house to sell it. If the price was underestimated, a buyer may be upset and unable to purchase the house when they find out must pay more than they expected. However, if the price was overestimated, the buyer will be able to purchase the house regardless.

### Step 4: Model Tuning

Features added:
* Square of LotArea
**Why:** Linear models can’t naturally fit curves, so squaring helps capture size-price relationships.
**Impact:** Slight improvement to RMSLE.

* Categorical feature interaction: LotArea * Neighborhood
**Why:** Captures how the value of land depends on neighborhood context.
**Impact:** Improvement to RMSLE

---

## 📊 Results & Takeaways

### ✅ Best Submission:
- **Model:** Ridge Regression with binarized categorical variables, zero-imputation, non-linear feature and feature interaction
- **Score:** `0.12984 RMSLE`  
- **Leaderboard Rank:** Top 2% globally (at time of submission)

### 🚀 Key Learnings:
- Smart feature binarization significantly improves model performance
- Ridge regularization helps reduce overfitting in high-dimensional spaces  
- NA values must be handled thoughtfully to avoid misleading patterns
- Conceptualize features in a real-world context to inform model tuning

---

## 💬 Final Thoughts

This project challenged me to think critically about data representation, interpretability, and generalization. By iterating through preprocessing strategies and model designs, I achieved a leaderboard score that placed me in the **top 2% globally**, making this one of my proudest machine learning milestones so far.

---

📩 **Feel free to reach out** if you have any questions!

