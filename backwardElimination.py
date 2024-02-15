import pandas as pd
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#"""2. Data Acquistion"""
dataset = pd.read_csv('50_Startups.csv')

#"""3.  Creating Data Frames
# Creating Data Frames


#dataset.apply(preprocessing.LabelEncoder().fit_transform(dataset["State"]))

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
dataset["State"]=labelencoder.fit_transform(dataset.State.values)


X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

X = sm.add_constant(dataset[['R&D Spend', 'Administration', 'Marketing Spend','State']])

X = pd.get_dummies(X, columns=["State"], drop_first=True)

"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)



# Fit the multiple linear regression model
model = sm.OLS(Y_train, X_train).fit()

# Print the summary which includes p-values for each coefficient
print(model.summary())

Y_pred = model.predict(X_test)

from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))

# Plotting each attribute against Profit
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# R&D Spend vs. Profit
axes[0, 0].scatter(X_train['R&D Spend'], Y_train, color='blue')
axes[0, 0].set_title('R&D Spend vs. Profit')
axes[0, 0].set_xlabel('R&D Spend')
axes[0, 0].set_ylabel('Profit')

# Administration vs. Profit
axes[0, 1].scatter(X_train['Administration'], Y_train, color='red')
axes[0, 1].set_title('Administration vs. Profit')
axes[0, 1].set_xlabel('Administration')
axes[0, 1].set_ylabel('Profit')

# Marketing Spend vs. Profit
axes[1, 0].scatter(X_train['Marketing Spend'], Y_train, color='green')
axes[1, 0].set_title('Marketing Spend vs. Profit')
axes[1, 0].set_xlabel('Marketing Spend')
axes[1, 0].set_ylabel('Profit')

# State vs. Profit
axes[1, 1].scatter(X_train['State_2'], Y_train, color='purple')
axes[1, 1].set_title('State vs. Profit')
axes[1, 1].set_xlabel('State')
axes[1, 1].set_ylabel('Profit')

plt.tight_layout()
plt.show()
----Without State---------------------------------

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#"""2. Data Acquistion"""
dataset = pd.read_csv('50_Startups.csv')

X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]


# Add a constant term for the intercept in the regression model
X = sm.add_constant(dataset[['R&D Spend', 'Administration', 'Marketing Spend']])




"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)



# Fit the multiple linear regression model
model = sm.OLS(Y_train, X_train).fit()

# Print the summary which includes p-values for each coefficient
print(model.summary())

Y_pred = model.predict(X_test)

from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))



----Without Administration ---------------------------------

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#"""2. Data Acquistion"""
dataset = pd.read_csv('50_Startups.csv')

X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]


# Add a constant term for the intercept in the regression model
X = sm.add_constant(dataset[['R&D Spend', 'Marketing Spend']])




"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)



# Fit the multiple linear regression model
model = sm.OLS(Y_train, X_train).fit()

# Print the summary which includes p-values for each coefficient
print(model.summary())

Y_pred = model.predict(X_test)

from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))



----Without Marketing Spend  ---------------------------------

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#"""2. Data Acquistion"""
dataset = pd.read_csv('50_Startups.csv')

X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]


# Add a constant term for the intercept in the regression model
X = sm.add_constant(dataset[['R&D Spend']])




"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)



# Fit the multiple linear regression model
model = sm.OLS(Y_train, X_train).fit()

# Print the summary which includes p-values for each coefficient
print(model.summary())

Y_pred = model.predict(X_test)

from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))



----Best Model With 'R&D Spend', 'Marketing Spend' ---------------------------------

import pandas as pd
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

#"""2. Data Acquistion"""
dataset = pd.read_csv('50_Startups.csv')

X = dataset.drop("Profit", axis=1)
y = dataset["Profit"]


# Add a constant term for the intercept in the regression model
X = sm.add_constant(dataset[['R&D Spend', 'Marketing Spend']])




"""6. Dataset Split"""
#train test split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size= 0.2, random_state = 0)



# Fit the multiple linear regression model
model = sm.OLS(Y_train, X_train).fit()

# Print the summary which includes p-values for each coefficient
print(model.summary())

Y_pred = model.predict(X_test)

from sklearn import metrics
# Evaluating the model and printing the value of MAE
print('Mean Absolute Error')
print(metrics.mean_absolute_error(Y_test, Y_pred))
# Evaluating the model and printing the value of MSE
print('Mean Square Error')
print(metrics.mean_squared_error(Y_test, Y_pred))
# Evaluating the model and printing the value of RMSE
print('Root Mean Square Error')
print(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))


