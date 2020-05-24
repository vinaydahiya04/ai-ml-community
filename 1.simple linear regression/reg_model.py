import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_1 = "Linear_X_Test.csv"
file_2 = "Linear_Y_Train.csv"
file_3 = "Linear_X_Train.csv"

X_train = pd.read_csv(file_3)
Y_train = pd.read_csv(file_2)
X_test = pd.read_csv(file_1)

from sklearn.linear_model import LinearRegression

reg_model = LinearRegression()
reg_model.fit(X_train,Y_train)

Y_pred = reg_model.predict(X_test)

file_4 = "Linear_Y_Test.csv"
df = pd.DataFrame(Y_pred)
df.rename(columns = {0:"Y"},inplace = True)
df.to_csv(file_4)

plt.scatter(x = X_train, y = Y_train, color =  "green")
plt.plot(X_train, reg_model.predict(X_train), color = "black")
plt.title("(Training Set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.scatter(x = X_test, y = Y_pred, color =  "red")
plt.plot(X_train, reg_model.predict(X_train), color = "black")
plt.title("(Test Set)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


