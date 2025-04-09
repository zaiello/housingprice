# 🏡 House Price Prediction with Regression Models

## 📌 Objective

Build a high-performance regression model to predict home sale prices using the **Ames Housing Dataset**. This project involved feature engineering, transformation, and missing value imputation in Python.

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

### 👷‍♀️ Step 1: Feature Engineering

Machine learning models (especially linear ones like Linear Regression, k-NN, etc.) operate on numerical inputs. If we were to leave the categories as strings or even assign them numbers arbitrarily (e.g., Gable = 1, Hip = 2, Flat = 3), the model would mistakenly interpret a relationship or order between them — like "Hip" being greater than "Gable", which doesn’t make any logical sense. Naïve binarization (binarizing all features) assumes that all features are independent, which isn’t true in our dataset (ex. LotArea is positively correlated with SalePrice). Since the assumption of independence is not met, the accuracy of our predictions can be negatively affected. Naïve binarization also doesn’t handle NA (or missing) values differently from actual features which leads to data sparseness: using missing data to make predictions affects their accuracy. In addition to reducing the accuracy of our predictions, the use of missing data wastes space and computational power.

So we binarize:

✅ To make the data numeric so models can process it

✅ To prevent the model from making erroneous assumptions about relationships between categorical values

✅ To ensure each category has equal weight and independence in the model

✅ To ensure missing data are handled appropriately 


> LotFrontage and GarageYrBlt are mixed fields - they have both numeric and categorical elements
> These fields were treated as numeric to preserve their order and scale.



### 🔧 Step 2: Missing Value Imputation 

* Default Strategy: Zero-imputation
* Tested Alternatives: Mean and median imputation

> Median/mean imputation increased RMSLE — possibly due to disrupting regularization balance or feature distributions.



### 📐 Step 3: Fit a Ridge Regression Model on Log-Transformed Outcome Variable

🎯 Why Ridge?
* Regularization helps generalize when feature space is large (e.g., many binary columns)
* Shrinks correlated features → less overfitting
* More robust in sparse, noisy data

🎯 Why Log-Transform SalePrice?
* Aligns with RMSLE (log-scale loss)
* Reduces skewness & handles large price variance
* RMSLE penalizes underestimates more — ideal for pricing models

### Step 4: Model Tuning

| Feature Change/Addition       | Why?                                                             | Impact on RMSLE        |
|-------------------------------|------------------------------------------------------------------|------------------------|
| ➕ `LotArea²`                  | Models non-linear relationship between size and price            | ✅ Slight improvement   |
| ➕ `LotArea * Neighborhood`    | Captures neighborhood-context in lot area's price influence      | ⭐ Big improvement     |


---

## 📊 Results & Takeaways

### ✅ Best Submission:
- **Model:** Ridge Regression with:
  - Smart feature binarization
  - Zero-imputation
  - Non-linear and interaction features
- **Dev Score:** `0.12516 RMSLE`  
- **Test Score:** `0.12984 RMSLE`  
- **Kaggle Rank:** 🏆 Top 2%

### 🚀 Key Learnings:
- Smart feature binarization significantly improves model performance
- Ridge regularization helps reduce overfitting in high-dimensional spaces  
- NA values must be handled thoughtfully to avoid misleading patterns
- Conceptualize features in a real-world context to inform model tuning

---

## 💬 Final Thoughts

This project challenged me to think critically about data representation, interpretability, and generalization. By iterating through preprocessing strategies and model designs, I achieved a leaderboard score that placed me in the **top 2%** of submissions, making this one of my proudest machine learning milestones so far.

---

📩 **Feel free to reach out** if you have any questions!

