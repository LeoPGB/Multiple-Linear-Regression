#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("multiple_linear.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1].values

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.5)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
df = pd.DataFrame({'Real Value':y_test, 'Predict Value': y_pred})
print(df)

print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# %%