import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")

print(df.info())

X = df.drop(columns=["risk_score", 'risk_class'])
y = df[['risk_score']]

x_train, x_test, y_train, y_test = train_test_split(X, y , test_size= 0.8, random_state= 6)

mlr_model = LinearRegression()

mlr_model.fit(x_train, y_train)

mlr_score = mlr_model.score(x_test, y_test)
print(mlr_score)

