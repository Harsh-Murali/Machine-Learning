import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# read and preprocess that dataset
train_calc = pd.read_csv('calc_case_description_train_set.csv')
test_calc = pd.read_csv('calc_case_description_test_set.csv')
train_mass = pd.read_csv('mass_case_description_train_set.csv')
test_mass = pd.read_csv('mass_case_description_test_set.csv')

# combine train and test datasets
calc = pd.concat([train_calc, train_calc], axis = 0)
mass = pd.concat([train_mass, test_mass], axis=0)

mean_calc = calc.mean()
std_calc = calc.std()

mean_mass = mass.mean()
std_mass = mass.std()

calc.keys()

mass.keys()

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
        data['pathology'] = le_pathology.fit_transform(data['pathology'])
        
        # rename columns
        data.rename(columns={'abnormality id': 'number of abnormality', 
                             'assessment' : 'overall BI-RADS assessment'}, inplace=True)
        try:
                data.rename(columns={'breast_density' : 'breast density'}, inplace=True)
        except KeyError:
                pass

        # Split the data back into train and test datasets
        return train_test_split(data, test_size=0.2, random_state=42)


# Split the data back into train and test datasets
train_calc, test_calc = preprocess(calc)
train_mass, test_mass = preprocess(mass)
    
# Split the data back into train and test datasets
train_calc, test_calc = preprocess(calc)
train_mass, test_mass = preprocess(mass)

train_calc[:3]

train_mass[:3]


# # DATASETS & DATALOADERS

# dataset and dataloader for calcification dataset
class BreastCancerDatasetCalc(Dataset):
    def __init__(self, df):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        features = torch.tensor([row['breast_density'], row['number_of_abnormality'],
                                row['calcification_type'], row['calcification_distribution'],
                                row['subtlety_rating']])
        overall_BI_RADS_assessment = torch.tensor(row['overall_BI-RADS_assessment'])
        pathology = torch.tensor(row['pathology'])
        
        return features, overall_BI_RADS_assessment, pathology

train_calc_dataset = BreastCancerDatasetCalc(train_calc)
test_calc_dataset = BreastCancerDatasetCalc(test_calc)

train_calc_loader = DataLoader(train_calc_dataset, batch_size=32, shuffle=True)
test_calc_loader = DataLoader(test_calc_dataset, batch_size=32, shuffle=False)

# dataset and dataloader for mass dataset
class BreastCancerDatasetMass(Dataset):
    def __init__(self, df):
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        features = torch.tensor([row['breast_density'], row['number_of_abnormality'],
                                row['mass_shape'], row['mass_margins'],
                                row['subtlety_rating']])
        overall_BI_RADS_assessment = torch.tensor(row['overall_BI-RADS_assessment'])
        pathology = torch.tensor(row['pathology'])
        
        return features, overall_BI_RADS_assessment, pathology

train_mass_dataset = BreastCancerDatasetMass(train_mass)
test_mass_dataset = BreastCancerDatasetMass(test_mass)

train_mass_loader = DataLoader(train_mass_dataset, batch_size=32, shuffle=True)
test_mass_loader = DataLoader(test_mass_dataset, batch_size=32, shuffle=False)

# # Transforms

# Create a custom transform function to add random noise to the features
class AddRandomNoise(object):
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def __call__(self, features):
        noise = torch.tensor(np.random.normal(0, self.noise_std, features.shape), dtype=torch.float32)
        noisy_features = features + noise
        return noisy_features

# Update the BreastCancerDatasetCalc class to accept multiple transforms
class BreastCancerDatasetCalc(Dataset):
    def __init__(self, df, transforms=None):
        self.data = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        features = torch.tensor([row['breast_density'], row['number_of_abnormality'],
                                row['calcification_type'], row['calcification_distribution'],
                                row['subtlety_rating']])
        
        if self.transforms:
            for transform in self.transforms:
                features = transform(features)
        
        overall_BI_RADS_assessment = torch.tensor(row['overall_BI-RADS_assessment'])
        pathology = torch.tensor(row['pathology'])
        
        return features, overall_BI_RADS_assessment, pathology

# Update the BreastCancerDatasetMass class to accept multiple transforms
class BreastCancerDatasetMass(Dataset):
    def __init__(self, df, transforms=None):
        self.data = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        features = torch.tensor([row['breast_density'], row['number_of_abnormality'],
                                row['mass_shape'], row['mass_margins'],
                                row['subtlety_rating']])
        
        if self.transforms:
            for transform in self.transforms:
                features = transform(features)
        
        overall_BI_RADS_assessment = torch.tensor(row['overall_BI-RADS_assessment'])
        pathology = torch.tensor(row['pathology'])
        
        return features, overall_BI_RADS_assessment, pathology

# Create instances of NormalizeFeatures and AddRandomNoise, then pass them to the dataset
class NormalizeFeatures(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, features):
        normalized_features = (features - self.mean) / self.std
        return normalized_features

normalize_transform = NormalizeFeatures(mean_calc, std_calc)
noise_transform = AddRandomNoise(noise_std=0.1)

train_calc_dataset = BreastCancerDatasetCalc(train_calc, transforms=[normalize_transform, noise_transform])
test_calc_dataset = BreastCancerDatasetCalc(test_calc, transforms=[normalize_transform]) # Do not apply noise to test data

train_calc_loader = DataLoader(train_calc_dataset, batch_size=32, shuffle=True)
test_calc_loader = DataLoader(test_calc_dataset, batch_size=32, shuffle=False)

normalize_transform = NormalizeFeatures(mean_mass, std_mass)
noise_transform = AddRandomNoise(noise_std=0.1)

train_mass_dataset = BreastCancerDatasetCalc(train_mass, transforms=[normalize_transform, noise_transform])
test_mass_dataset = BreastCancerDatasetCalc(test_mass, transforms=[normalize_transform]) # Do not apply noise to test data

train_mass_loader = DataLoader(train_mass_dataset, batch_size=32, shuffle=True)
test_mass_loader = DataLoader(test_mass_dataset, batch_size=32, shuffle=False)


# define the KNN model

features_calc = ['breast density', 'number of abnormality', 'calc type', 'calc distribution', 'subtlety']

y_train_calc = train_calc['overall BI-RADS assessment']
x_train_calc = train_calc[features_calc]
y_test_calc = test_calc['overall BI-RADS assessment']
x_test_calc = test_calc[features_calc]


model_calc = KNeighborsClassifier(n_neighbors=20)
model_calc.fit(x_train_calc, y_train_calc)
y_pred_calc = model_calc.predict(x_test_calc)
mse_calc = mean_squared_error(y_test_calc, y_pred_calc)
score_calc = model_calc.score(x_test_calc, y_test_calc)

print(classification_report(y_test_calc, y_pred_calc))
print(confusion_matrix(y_test_calc, y_pred_calc))
print('Mean Squared Error:', mse_calc)
print('Model Score:', score_calc)
print()


features_mass = ['breast density', 'number of abnormality', 'mass shape', 'mass margins', 'subtlety']

y_train_mass = train_mass['overall BI-RADS assessment']
x_train_mass = train_mass[features_mass]
y_test_mass = test_mass['overall BI-RADS assessment']
x_test_mass = test_mass[features_mass]

model_mass = KNeighborsClassifier(n_neighbors=5)
model_mass.fit(x_train_mass, y_train_mass)
y_pred_mass = model_mass.predict(x_test_mass)
mse_mass = mean_squared_error(y_test_mass, y_pred_mass)
score_mass = model_mass.score(x_test_mass, y_test_mass)

print(classification_report(y_test_mass, y_pred_mass, zero_division=1))
print(confusion_matrix(y_test_mass, y_pred_mass))
print('Mean Squared Error:', mse_mass)
print('Model Score:', score_mass)
print()