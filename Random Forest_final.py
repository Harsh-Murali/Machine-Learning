# Import relevant libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix

# Import four datasets
train_calc = pd.read_csv(r'C:\Users\Harsh\Downloads\calc_case_description_train_set.csv')
test_calc = pd.read_csv(r'C:\Users\Harsh\Downloads\calc_case_description_test_set.csv')
train_mass = pd.read_csv(r'C:\Users\Harsh\Downloads\mass_case_description_train_set.csv')
test_mass = pd.read_csv(r'C:\Users\Harsh\Downloads\mass_case_description_test_set.csv')

original = {'train_calc': train_calc, 'test_calc': test_calc, 
        'train_mass': train_mass, 'test_mass': test_mass}

calc = train_calc.values.tolist() + test_calc.values.tolist()
calc = pd.DataFrame(calc, columns = train_calc.columns)
mass = train_mass.values.tolist() + test_mass.values.tolist()
mass = pd.DataFrame(mass, columns = train_mass.columns)

# Preprocess datasets
def preprocess(data):
        # make a copy of the data to avoid SettingWithCopyWarning
        data = data.copy()
        
        # set the limitations on the numerical columns
        try:
                data['breast density'] = data['breast density'].clip(1, 4)
        except KeyError:
                data['breast_density'] = data['breast_density'].clip(1, 4)
        data['abnormality id'] = data['abnormality id'].clip(0)
        data['assessment'] = data['assessment'].clip(0, 5)
        data['subtlety'] = data['subtlety'].clip(1, 5)
        
        # change the name of index
        data.index = data['patient_id'] + '_' + data['image view'] + '_' \
        + data['left or right breast'] + '_' + data['abnormality id'].astype(str)

        # Remove useless columns
        data = data[data.columns.drop(list(data.filter(regex='file path')) 
                + ['image view', 'patient_id', 'left or right breast', 'abnormality type'])]

        # Fill NaN values with appropriate placeholders
        try:
                data['calc type'] = data['calc type'].fillna('None')
                data['calc distribution'] = data['calc distribution'].fillna('None')
        except KeyError:
                data['mass shape'] = data['mass shape'].fillna('None')
                data['mass margins'] = data['mass margins'].fillna('None')

        '''
        pathology :
        BENIGN_WITHOUT_CALLBACK = 0
        BENIGN = 0.5
        MALIGNANT = 1
        '''
        data['pathology'] = data['pathology'].map({'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 1, 'MALIGNANT': 2})
        

        # Encode categorical features
        le_pathology = LabelEncoder()
        try:
                le_type = LabelEncoder()
                le_distribution = LabelEncoder()
                
                data['calc type'] = le_type.fit_transform(data['calc type'])
                data['calc distribution'] = le_distribution.fit_transform(data['calc distribution'])
                
        except KeyError:
                le_shape = LabelEncoder()
                le_distribution = LabelEncoder()
                
                data['mass shape'] = le_shape.fit_transform(data['mass shape'])
                data['mass margins'] = le_distribution.fit_transform(data['mass margins'])
        
        # rename columns
        data.rename(columns={'abnormality id': 'number of abnormalities', 
                             'assessment' : 'overall BI-RADS assessment'}, inplace=True)
        try:
                data.rename(columns={'breast_density' : 'breast density'}, inplace=True)
                # split
                return data[:train_mass.shape[0]], data[train_mass.shape[0]:]
        except KeyError:
                return data[:train_calc.shape[0]], data[train_calc.shape[0]:]
            
train_calc, test_calc = preprocess(calc)
train_mass, test_mass = preprocess(mass)



# Perform grid search using random forest on calcification dataset to predict assessment

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix
import pandas as pd

# Define relevant features for the model
features = ['breast density', 'number of abnormalities', 'calc type', 'calc distribution', 'subtlety']

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

# Create a random forest model object
rf = RandomForestRegressor()

# Create a GridSearchCV object to perform the search
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='r2')

# Train the model on the calcification dataset
test_X = test_calc[features]
test_Y = test_calc['overall BI-RADS assessment']
train_X = train_calc[features]
train_Y = train_calc['overall BI-RADS assessment']

grid_search.fit(train_X, train_Y)

# Print out the best hyperparameters and score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Make predictions on the test set using the best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(test_X)

# Print a confusion matrix to evaluate performance
conf_m = confusion_matrix(test_Y, y_pred.round())
print("Confusion Matrix:\n", conf_m)

# Calculate the accuracy of predictions
accuracy = accuracy_score(test_Y, y_pred.round())
print('Accuracy:', accuracy)

# Calculate the Mean Squared Error of predictions
mse = mean_squared_error(test_Y, y_pred)
print('Mean Squared Error:', mse)

# Print out feature importances
importances = best_rf.feature_importances_
sorted_indices = importances.argsort()[::-1]
print('Feature importances:')
for index in sorted_indices:
    print(f'{features[index]}: {importances[index]}')
    

    
# Perform grid search using random forest on mass dataset to predict assessment
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
import pandas as pd

# Define relevant features for the model
features = ['breast density', 'number of abnormalities', 'mass shape', 'mass margins', 'subtlety']

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

# Create a random forest model object
rf = RandomForestRegressor()

# Create a GridSearchCV object to perform the search
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='r2')

# Train the model on the mass dataset
train_X = train_mass[features]
train_Y = train_mass['overall BI-RADS assessment']
test_X = test_mass[features]
test_Y = test_mass['overall BI-RADS assessment']

grid_search.fit(train_X, train_Y)

# Print out the best hyperparameters and score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# Predict the test set using the current best model
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(test_X)

# Print a confusion matrix to evaluate performance
conf_m = confusion_matrix(test_Y, y_pred.round())
print("Confusion Matrix:\n", conf_m)

# Calculate the accuracy of predictions
accuracy = accuracy_score(test_Y, y_pred.round())
print('Accuracy:', accuracy)

# Calculate the Mean Squared Error of predictions
mse = mean_squared_error(test_Y, y_pred)
print('Mean Squared Error:', mse)

# Print out feature importances
importances = best_rf.feature_importances_
sorted_indices = importances.argsort()[::-1]
print('Feature importances:')
for index in sorted_indices:
    print(f'{features[index]}: {importances[index]}')

            

# Perform grid search using random forest on calcification dataset to predict pathology using assessment and relevant features

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import pandas as pd

# Define relevant features including assessment for the model
features = ['breast density', 'number of abnormalities', 'calc type', 'calc distribution', 'subtlety', 'overall BI-RADS assessment']

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

# Create a random forest model object
rf = RandomForestRegressor()

# Create a GridSearchCV object for performing the search
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='r2')

# Train the model using the calcification dataset
train_X = train_calc[features]
train_Y = train_calc['pathology']

grid_search.fit(train_X, train_Y)

# Print the best hyperparameters and the bestscore
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

best_rf = grid_search.best_estimator_

# Make predictions using the test set
test_X = test_calc[features]
test_Y = test_calc['pathology']
test_pred = best_rf.predict(test_X)

# Create a confusion matrix
conf_m = confusion_matrix(test_Y, test_pred.round())
print('Confusion matrix:')
print(conf_m)

# Calculate the accuracy of predictions
accuracy = accuracy_score(test_Y, test_pred.round())
print('Accuracy:', accuracy)

# Calculate the Mean Square Error of prediction
mse = mean_squared_error(test_Y, test_pred)
print('Mean Squared Error:', mse)

# Print out feature importances of prediction
importances = best_rf.feature_importances_
sorted_indices = importances.argsort()[::-1]
print('Feature importances:')
for index in sorted_indices:
    print(f'{features[index]}: {importances[index]}')



# GRID SEARCH WITH RANDOM FOREST MASS                            BEST SCORE
# Perform grid search using random forest on mass dataset to predict pathology using assessment and relevant features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import pandas as pd

# Define relevant features including assessment for the model
features = ['breast density', 'number of abnormalities', 'mass shape', 'mass margins', 'subtlety', 'overall BI-RADS assessment']

# Define a parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state': [42]
}

# Create a random forest model object
rf = RandomForestRegressor()

# Create a GridSearchCV object to perform the search on
grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, scoring='r2')

# Train the model using the mass dataset
train_X = train_mass[features]
train_Y = train_mass['pathology']

grid_search.fit(train_X, train_Y)

# Print out the best hyperparameters and score
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

best_rf = grid_search.best_estimator_

# Make predictions using the test set
test_X = test_mass[features]
test_Y = test_mass['pathology']
test_pred = best_rf.predict(test_X)

# Create a confusion matrix
conf_m = confusion_matrix(test_Y, test_pred.round())
print('Confusion matrix:')
print(conf_m)

# Calculate the accuracy of predictions
accuracy = accuracy_score(test_Y, test_pred.round())
print('Accuracy:', accuracy)

# Calculate the Mean Squared Error of predictions
mse = mean_squared_error(test_Y, test_pred)
print('Mean Squared Error:', mse)

# Print out the feature importances
importances = best_rf.feature_importances_
sorted_indices = importances.argsort()[::-1]
print('Feature importances:')
for index in sorted_indices:
    print(f'{features[index]}: {importances[index]}')


            
            
            
            
            