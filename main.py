import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("Housing.csv")

X = df.drop("price", axis=1)
y = df["price"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

if rf_r2 > lr_r2:
    final_model = rf
    final_name = "RandomForestRegressor"
else:
    final_model = lr
    final_name = "LinearRegression"

joblib.dump(
    {
        "model": final_model,
        "features": X.columns.tolist()
    },
    "final_model_bundle.pkl"
)

print("Linear Regression R2:", lr_r2)
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest R2:", rf_r2)
print("Random Forest RMSE:", rf_rmse)
print("Final Model Selected:", final_name)
print("final_model_bundle.pkl created successfully")
