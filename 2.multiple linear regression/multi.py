import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr


file1 = "Train.csv"
file2 = "Test.csv"

train = pd.read_csv(file1)

train_x = train.iloc[: , :-1]
train_y = train.iloc[:, -1:]

test_x = pd.read_csv(file2)

#on checking the pearson coefficient i come to the conclusion that all 5 features need to be used for training the model

from sklearn.linear_model import LinearRegression
model  = LinearRegression()
model.fit(train_x,train_y)
test_y = model.predict(test_x)

yf = pd.DataFrame(test_y)
file4 = "Test_Y.csv"
yf.to_csv(file4)



