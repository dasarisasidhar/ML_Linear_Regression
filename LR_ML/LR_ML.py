import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn import metrics


USAhousing = pd.read_csv('../dataset/USAHousing.csv')
USAhousing.head()
USAhousing.info()
USAhousing.describe()
USAhousing.columns
plot = sns.pairplot(USAhousing) #get all the pair plots
sns.distplot(USAhousing['Price']) #get dist plots
sns.heatmap(USAhousing.corr()) #get he4atmap
X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']] #input
y = USAhousing['Price'] #output to be calculates
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#divide the data to test and train randomply (80% train and 20% test)
lr = LinearRegression()
lr.fit(X_train,y_train) #train the data
coeff_df = pd.DataFrame(lr.coef_,X.columns,columns=['Coefficient'])
boston = load_boston()
print(boston.DESCR)
boston_df = boston.data
predictions = lr.predict(X_test)
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),bins=50);
print('MAE:', metrics.mean_absolute_error(y_test, predictions)) 
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

"""
Regression Evaluation Metrics
Here are three common evaluation metrics for regression problems:

Mean Absolute Error (MAE) is the mean of the absolute value of the errors:

1𝑛∑𝑖=1𝑛|𝑦𝑖−𝑦̂ 𝑖|
 
Mean Squared Error (MSE) is the mean of the squared errors:

1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2
 
Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:

1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
 
Comparing these metrics:

MAE is the easiest to understand, because it's the average error.
MSE is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
RMSE is even more popular than MSE, because RMSE is interpretable in the "y" units.
All of these are loss functions, because we want to minimize them.
"""