# Import the libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Import the dataset
train_calc = pd.read_csv('calc_case_description_train_set.csv') # (1546, 14)
test_calc = pd.read_csv('calc_case_description_test_set.csv') # (326, 14)
train_mass = pd.read_csv('mass_case_description_train_set.csv') # (1318, 14)
test_mass = pd.read_csv('mass_case_description_test_set.csv') # (378, 14)

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
        data['pathology'] = data['pathology'].map({'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 0.5, 'MALIGNANT': 1})
        

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
        except KeyError:
                pass

        return data
    
calc = preprocess(calc)
mass = preprocess(mass)
train_calc, test_calc = calc[:train_calc.shape[0]], calc[train_calc.shape[0]:]
train_mass, test_mass = mass[:train_mass.shape[0]], mass[train_mass.shape[0]:]

# change the pathology to discrete values (0, 1, 2)
label = LabelEncoder()
train_calc['pathology'] = label.fit_transform(train_calc['pathology'])
test_calc['pathology'] = label.fit_transform(test_calc['pathology'])
train_mass['pathology'] = label.fit_transform(train_mass['pathology'])
test_mass['pathology'] = label.fit_transform(test_mass['pathology'])

import sklearn.tree as tree
import graphviz

# Decision Tree with no pruning
assessment_calc = tree.DecisionTreeClassifier(criterion='entropy')
assessment_calc = assessment_calc.fit(train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_calc['overall BI-RADS assessment'])

print('Score Against Training Data: ', assessment_calc.score(train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_calc['overall BI-RADS assessment']))
print('Score Against Test Data: ', assessment_calc.score(test_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), test_calc['overall BI-RADS assessment']))

# print tree information
print("Number of Leaves: ", assessment_calc.get_n_leaves())
print("Depth of Tree: ", assessment_calc.get_depth())

# export the tree to a pdf file
graph = tree.export_graphviz(assessment_calc, out_file=None, feature_names=train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1).columns, class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(graph)
graph.render('calc_tree')

# Decision Tree with pruning
assessment_calc_pruned = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10, max_depth=5)
assessment_calc_pruned = assessment_calc_pruned.fit(train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_calc['overall BI-RADS assessment'])

print('Score Against Training Data: ', assessment_calc_pruned.score(train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_calc['overall BI-RADS assessment']))
print('Score Against Test Data: ', assessment_calc_pruned.score(test_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1), test_calc['overall BI-RADS assessment']))

# export the tree to a pdf file
graph = tree.export_graphviz(assessment_calc_pruned, out_file=None, feature_names=train_calc.drop(['pathology', 'overall BI-RADS assessment'], axis=1).columns, class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(graph)
graph.render('calc_tree_pruned')

# Decision Tree without subtlety
assessment_calc_no_subtlety = tree.DecisionTreeClassifier(criterion='entropy')
assessment_calc_no_subtlety = assessment_calc_no_subtlety.fit(train_calc.drop(['pathology', 'overall BI-RADS assessment', 'subtlety'], axis=1), train_calc['overall BI-RADS assessment'])

print('Score Against Training Data: ', assessment_calc_no_subtlety.score(train_calc.drop(['pathology', 'overall BI-RADS assessment', 'subtlety'], axis=1), train_calc['overall BI-RADS assessment']))
print('Score Against Test Data: ', assessment_calc_no_subtlety.score(test_calc.drop(['pathology', 'overall BI-RADS assessment', 'subtlety'], axis=1), test_calc['overall BI-RADS assessment']))

#Mass Models for 'overall BI-RADS assessment'
assessment_mass = tree.DecisionTreeClassifier(criterion='entropy')
assessment_mass = assessment_mass.fit(train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_mass['overall BI-RADS assessment'])

print('Score Against Training Data: ', assessment_mass.score(train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_mass['overall BI-RADS assessment']))
print('Score Against Test Data: ', assessment_mass.score(test_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), test_mass['overall BI-RADS assessment']))

# export the tree to a pdf file
graph = tree.export_graphviz(assessment_mass, out_file=None, feature_names=train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1).columns, class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(graph)
graph.render('mass_tree')

# Decision Tree with pruning
assessment_mass_pruned = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
assessment_mass_pruned = assessment_mass_pruned.fit(train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_mass['overall BI-RADS assessment'])

print('Score Against Training Data: ', assessment_mass_pruned.score(train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), train_mass['overall BI-RADS assessment']))
print('Score Against Test Data: ', assessment_mass_pruned.score(test_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1), test_mass['overall BI-RADS assessment']))

# export the tree to a pdf file
graph = tree.export_graphviz(assessment_mass_pruned, out_file=None, feature_names=train_mass.drop(['pathology', 'overall BI-RADS assessment'], axis=1).columns, class_names=['1', '2', '3', '4', '5'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(graph)
graph.render('mass_tree_pruned')

model = tree.DecisionTreeClassifier()
model.fit(train_calc.drop('pathology', axis=1), train_calc['pathology'])

model.score(test_calc.drop('pathology', axis=1), test_calc['pathology'])

# export the tree to a pdf file
data = tree.export_graphviz(model, out_file=None, feature_names=train_calc.drop('pathology', axis=1).columns, class_names=['benign without callback', 'benign', 'malignant'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(data)
graph.render("calc_pathology_tree")

model = tree.DecisionTreeClassifier()
model.fit(train_mass.drop('pathology', axis=1), train_mass['pathology'])

model.score(test_mass.drop('pathology', axis=1), test_mass['pathology'])

# export the tree to a pdf file
data = tree.export_graphviz(model, out_file=None, feature_names=train_mass.drop('pathology', axis=1).columns, class_names=['benign without callback', 'benign', 'malignant'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(data)
graph.render("mass_pathology_tree")