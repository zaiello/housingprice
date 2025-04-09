'''Housing Price Prediction: Kaggle Competition'''
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import pandas as pd

# Import data
train_data = map(lambda s: s.strip().split(","), open('my_train.csv').readlines()[1:]) #skip the header (first row)
dev_data = map(lambda s: s.strip().split(","), open('my_dev.csv').readlines()[1:]) #skip the header (first row)
test_data = map(lambda s: s.strip().split(","), open('test.csv').readlines()[1:]) #skip the header (first row)

# get data as a df
df = pd.read_csv('my_train.csv')

# Find numerical columns
df_cols = df.select_dtypes(exclude=['object']).columns

numerical_fields = []
# Get indexes of numerical fields
for c in df_cols:
    numerical_fields.append(df.columns.get_loc(c))
    # print(f'{c}: {df.columns.get_loc(c)}')


# Set up training data
mapping = {}

train_targets = []
new_train_categorical_data = []
new_train_numerical_data = []

for row in train_data:
    numerical_row = []
    categorical_row = []
    # add a field for the combination of LotArea (4) and Neighborhood(12)
    row.append(row[4] + row[12])
    # add one for the square of LotArea
    row.append(np.square(float(row[4])))
    for j, x in enumerate(row):
        if j == 0:
             # skip "Id" column, it is not a feature.
            continue
        feature = (j, x) #j is the index in this row
        if j == 80:  # target index
            train_targets.append(float(x)) # For this homework, we need the real value targets.
        else:
            if j != 1 and j in numerical_fields:
                if x == 'NA':
                    x = 0
                    numerical_row.append(float(x))
                else:
                    numerical_row.append(float(x))
            else:
                # process categorical fields, assign an index for each feature
                if feature not in mapping:
                    mapping[feature] = len(mapping) # insert a new feature into the index
                categorical_row.append(mapping[feature])
    new_train_categorical_data.append(categorical_row)
    new_train_numerical_data.append(numerical_row)


# create the training matrix
bindata_train = np.zeros((len(new_train_categorical_data), len(mapping)))
for i, row in enumerate(new_train_categorical_data):
    for x in row:
        bindata_train[i][x] = 1

# there should be 38 numerical fields minus MSSubClass, ID, and SalePrice (38-3 = 35) plus one for the square of
# LotArea (len(numerical_fields) - 2)
numerical_train = np.zeros((len(new_train_numerical_data), len(numerical_fields)-2))

for i, row in enumerate(new_train_numerical_data):
    for j, x in enumerate(row):
        numerical_train[i][j] = x

bindata_train = np.concatenate((bindata_train, numerical_train), axis=1)

# Set up dev data
dev_targets = []
new_dev_categorical_data = []
new_dev_numerical_data = []
for row in dev_data:
    numerical_row = []
    categorical_row = []
    # add a field for the combination of LotArea (4) and Neighborhood(12)
    row.append(row[4] + row[12])
    # add one for the square of LotArea
    row.append(np.square(float(row[4])))
    for j, x in enumerate(row):
        if j == 0:
            # skip "Id" column, it is NOT a feature.
            continue
        feature = (j, x) #j is the index in this row
        if j == 80: # target index
            dev_targets.append(float(x))  # For this homework, we need the real value targets.
        else:
            if j != 1 and j in numerical_fields:
                if x == 'NA':
                    x = 0
                    numerical_row.append(float(x))
                else:
                    numerical_row.append(float(x))
            else:
                # process categorical fields, assign an index for each feature
                if feature in mapping:
                    categorical_row.append(mapping[feature])
    new_dev_categorical_data.append(categorical_row)
    new_dev_numerical_data.append(numerical_row)

# create the training matrix
bindata_dev = np.zeros((len(new_dev_categorical_data), len(mapping)))

for i, row in enumerate(new_dev_categorical_data):
    for x in row:
        bindata_dev[i][x] = 1

# there should be 38 numerical fields minus MSSubClass, ID, and SalePrice (38-3 = 35) plus 1 for LotArea^2
# (len(numerical_fields) - 2)
numerical_dev = np.zeros((len(new_dev_numerical_data), len(numerical_fields)-2))

for i, row in enumerate(new_dev_numerical_data):
    for j, x in enumerate(row):
        numerical_dev[i][j] = x

bindata_dev = np.concatenate((bindata_dev, numerical_dev), axis=1)


# RMSLE
def RMSLE(p, y):
    return np.sqrt(np.square(np.log(p+1) - np.log(y+1)).mean())

# choose alpha
clf = Ridge(alpha=19.8)

# Implement ridge regression
ridgereg = clf.fit(bindata_train, np.log(np.array(train_targets)))

# predict
dev_pred_ridge = ridgereg.predict(bindata_dev)

# find RMSLE
print(RMSLE(np.exp(dev_pred_ridge), np.array(dev_targets)))



'''Predict on Test Data'''

# set up
test_ids = []
new_test_numerical_data = []
new_test_categorical_data = []
for row in test_data:
    row.append(np.square(float(row[4])))
    # add a field for the combination of LotArea (4) and Neighborhood(12)
    row.append(row[4] + row[12])
    numerical_row = []
    categorical_row = []
    for j, x in enumerate(row):
        if j == 0:
            # skip "Id" column, it is NOT a feature.
            test_ids.append(x) # store the id for submission
            continue
        feature = (j, x) #j is the index in this row
        if j in numerical_fields and j != 1:
            if x == 'NA':
                x = 0
                numerical_row.append(float(x))
            else:
                numerical_row.append(float(x))
        else:
            # process categorical fields, assign an index for each feature
            if feature in mapping:
                categorical_row.append(mapping[feature])
    new_test_categorical_data.append(categorical_row)
    new_test_numerical_data.append(numerical_row)

bindata_test = np.zeros((len(new_test_categorical_data), len(mapping)))
for i, row in enumerate(new_test_categorical_data):
    for x in row:
        bindata_test[i][x] = 1

# set up numerical matrix
numerical_test = np.zeros((len(new_test_numerical_data), len(numerical_fields)-2))

for i, row in enumerate(new_test_numerical_data):
    for j, x in enumerate(row):
        numerical_test[i][j] = x

# concatenate numerical and categorical matrices
bindata_test = np.concatenate((bindata_test, numerical_test), axis=1)

# predict
test_pred_ridge = ridgereg.predict(bindata_test)
test_pred = np.exp(test_pred_ridge)

# Write to file
# with open("part43_prediction.csv", "w") as wf:
#     print("Id,SalePrice", file=wf)  # print header
#     for cid, pred in zip(test_ids, test_pred):
#         print(f"{cid},{pred}", file=wf)
