# imports

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# read_file

df = pd.read_csv('./datas/canada_per_capita_income.csv')

annual_income = df.per_capita_income_US
year = df.drop('per_capita_income_US', axis=1)

# make linearRegression

reg = linear_model.LinearRegression()
reg.fit(year.values, annual_income)

# show per capita income with blue line , show LinearRegression with red line
# show 2020 year with green dot
plt.plot(df['year'], df['per_capita_income_US'])
plt.plot(year, reg.predict(year.values), color='red')
plt.scatter(2020, reg.predict([[2020]]), color='green')
plt.show()

# return result
print(reg.predict([[2020]]))
# [41288.69409442] <- this is result in dollar
