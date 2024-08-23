from Preprocessed_data2 import *
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Data preprocessing - target is assessment
y_train_calc = train_calc['overall BI-RADS assessment']
x_train_calc = train_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_calc = test_calc['overall BI-RADS assessment']
x_test_calc = test_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)

y_train_mass = train_mass['overall BI-RADS assessment']
x_train_mass = train_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_mass = test_mass['overall BI-RADS assessment']
x_test_mass = test_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)

params = {
    'num_leaves': [10, 31, 100],
    'max_depth': [-1, 3, 5, 10],
    'min_child_samples': [5, 20, 50],
    'learning_rate': [0.1, 0.2]
}

print('Predicting assessment - CALC:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
print('\tUsing hyperparameters from gridsearch')

# Using Gridsearch to tune hyperparameters
model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid=params)
grid_search.fit(x_train_calc, y_train_calc)
print('Maximum accuracy for predicting calc assessment:', grid_search.best_score_)
print('The parameters for this model are: ', grid_search.best_params_)

# Best model given by gridsearch
model = lgb.LGBMClassifier(learning_rate=0.2, max_depth=5, min_child_samples=50, num_leaves=31)
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))

print('Predicting assessment - MASS:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
print('\tUsing hyperparameters from gridsearch')

# Using Gridsearch to tune hyperparameters
model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid=params)
grid_search.fit(x_train_mass, y_train_mass)
print('Maximum accuracy for predicting mass assessment:', grid_search.best_score_)
print('The parameters for this model are: ', grid_search.best_params_)

# Best model given by gridsearch
model = lgb.LGBMClassifier(learning_rate=0.1, max_depth=3, min_child_samples=20, num_leaves=10)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))

# Data preprocessing - target is pathology
y_train_calc = train_calc['pathology']
x_train_calc = train_calc.drop(['pathology'], axis=1)
y_test_calc = test_calc['pathology']
x_test_calc = test_calc.drop(['pathology'], axis=1)

y_train_mass = train_mass['pathology']
x_train_mass = train_mass.drop(['pathology'], axis=1)
y_test_mass = test_mass['pathology']
x_test_mass = test_mass.drop(['pathology'], axis=1)

print('Predicting pathology using assessment - CALC:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
print('\tUsing hyperparameters from gridsearch')

# Using Gridsearch to tune hyperparameters
model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid=params)
grid_search.fit(x_train_calc, y_train_calc)
print('Maximum accuracy for predicting calc pathology:', grid_search.best_score_)
print('The parameters for this model are: ', grid_search.best_params_)

# Best model given by gridsearch
model = lgb.LGBMClassifier(learning_rate=0.1, max_depth=3, min_child_samples=50, num_leaves=10)
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))

print('Predicting pathology using assessment - MASS:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
print('\tUsing hyperparameters from gridsearch')

# Using Gridsearch to tune hyperparameters
model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid=params)
grid_search.fit(x_train_mass, y_train_mass)
print('Maximum accuracy for predicting mass pathology:', grid_search.best_score_)
print('The parameters for this model are: ', grid_search.best_params_)

# Best model given by gridsearch
model = lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, min_child_samples=50, num_leaves=10)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))

