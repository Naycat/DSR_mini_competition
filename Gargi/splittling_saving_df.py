import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# careful with this one. It is heavier than it look
train = pd.read_csv("rossmann-store-sales/new_train.csv", index_col = 0, low_memory = False)

train["month"] = pd.DatetimeIndex(train["Date"]).month
data = pd.get_dummies(train.drop("Date", axis = 1))

#put future files to later, and past files first
#so that the time series split makes sense
data = data.reindex(index = data.index[::-1], copy = False)
data = data.reset_index(drop = True)

tscv = TimeSeriesSplit(n_splits = 3)
idx1, idx2, idx3 = tscv.split(data)

train_idx, test_idx = idx3[0], idx3[1]

train_data = data.loc[train_idx]
test_data = data.loc[test_idx]

train_data.to_csv("rossmann-store-sales/train_data.csv")
test_data.to_csv("rossmann-store-sales/test_data.csv")
