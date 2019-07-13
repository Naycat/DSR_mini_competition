# I am on numpy == 1.16.4

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

train_data = pd.read_csv("rossmann-store-sales/train_data.csv", index_col = 0)
test_data = pd.read_csv("rossmann-store-sales/test_data.csv", index_col = 0)

train_labels = train_data.loc[:, "Sales"]
test_labels = test_data.loc[:, "Sales"]

train_features = train_data.drop("Sales", axis = 1)
test_features = test_data.drop("Sales", axis = 1)

rf = RandomForestRegressor(n_estimators = 100, random_state = 123)
rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
abs_errors = abs(predictions - test_labels)
sqr_errors = (predictions - test_labels)**2

print('Mean Absolute Error:', round(np.mean(abs_errors), 2))
print('Mean Squared Error:', round(np.mean(sqr_errors), 2))

plt.figure(figsize = (12, 6))
plt.plot((test_labels.index - test_labels.index[0]), abs_errors)
plt.show()
