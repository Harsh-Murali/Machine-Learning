import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Import the dataset
train_calc = pd.read_csv('calc_case_description_train_set.csv')
test_calc = pd.read_csv('calc_case_description_test_set.csv')
train_mass = pd.read_csv('mass_case_description_train_set.csv')
test_mass = pd.read_csv('mass_case_description_test_set.csv')

original = {'train_calc': train_calc, 'test_calc': test_calc, 
        'train_mass': train_mass, 'test_mass': test_mass}

calc = train_calc.values.tolist() + test_calc.values.tolist()
calc = pd.DataFrame(calc, columns = train_calc.columns)
mass = train_mass.values.tolist() + test_mass.values.tolist()
mass = pd.DataFrame(mass, columns = train_mass.columns)

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

def preprocess2(data):
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
                + ['image view', 'patient_id', 'abnormality id', 'abnormality type'])]

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
        data.rename(columns={'assessment' : 'overall BI-RADS assessment'}, inplace=True)
        try:
                data.rename(columns={'breast_density' : 'breast density'}, inplace=True)
                # split
                return data[:train_mass.shape[0]], data[train_mass.shape[0]:]
        except KeyError:
                return data[:train_calc.shape[0]], data[train_calc.shape[0]:]

train_calc_2, test_calc_2 = preprocess(calc)
train_mass_2, test_mass_2 = preprocess(mass)
