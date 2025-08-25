# PRODIGY_ML_01

# 🏠 House Price Prediction using Linear Regression

## 📌 Project Overview
This project implements a **Linear Regression model** to predict house prices based on:
- Square footage (`GrLivArea`)
- Number of bedrooms (`BedroomAbvGr`)
- Number of bathrooms (`FullBath`)

The dataset used is from Kaggle’s competition:  
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

---

## 📊 Dataset
- File: **train.csv**
- Target variable: `SalePrice`
- Selected features: `GrLivArea`, `BedroomAbvGr`, `FullBath`

---

## ⚙️ Technologies Used
- Python  
- Pandas & NumPy (data handling)  
- Scikit-learn (Linear Regression, evaluation metrics)  
- Matplotlib (visualization)  

---

## 📜 Code
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("train.csv")

# Select features and target
X = data[["GrLivArea", "BedroomAbvGr", "FullBath"]]
y = data["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("R² Score:", r2)
print("RMSE:", rmse)

# Sample Predictions
pred_df = pd.DataFrame({"Actual": y_test.values[:10], "Predicted": y_pred[:10]})
print(pred_df)

# Visualization
plt.scatter(y_test, y_pred, alpha=0.6, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

## Results:-
Model Coefficients:

GrLivArea (Square Footage): +104.03

BedroomAbvGr (Bedrooms): -26,655.17

FullBath (Bathrooms): +30,014.32

Intercept: 52,261.75

R² Score: 0.6341 (~63% of price variation explained)

RMSE: 52,975.72 (average prediction error ≈ ₹53K)

## Sample Predictions:-
Actual Price	Predicted Price
154,500	       113,410.67
325,000	       305,081.88
115,000	       135,904.79
159,000	       205,424.68
315,500	       227,502.68


## Visualization:-
~ Scatter plot of Actual vs Predicted prices:

~ Red dashed line = perfect prediction

~ Blue dots = predicted results

~ Most points are near the line, but higher-priced houses are underestimated

## Conclusion:-
~ Square footage strongly influences house prices.

~ Bedrooms have a slightly negative correlation (possibly due to redundancy with square footage).

~ Bathrooms increase value significantly.

The model is simple but explains ~63% of price variation.
