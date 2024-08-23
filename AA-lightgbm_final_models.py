from Preprocessed_data2 import *
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

## Final models for lightgbm after hyperparameter tuning iteratiely and via gridsearch

# Target is assessment 
x_train_calc = train_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_train_calc = train_calc['overall BI-RADS assessment']
x_test_calc = test_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_calc = test_calc['overall BI-RADS assessment']

x_train_mass = train_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_train_mass = train_mass['overall BI-RADS assessment']
x_test_mass = test_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_mass = test_mass['overall BI-RADS assessment']

target_names_calc = ['Assessment 0', 'Assessment 2', 'Assessment 3', 'Assessment 4', 'Assessment 5']
target_names_mass = ['Assessment 0', 'Assessment 1', 'Assessment 2', 'Assessment 3', 'Assessment 4', 'Assessment 5']
target_names_path = ['Benign without callback', 'Benign', 'Malignant']

print('Predicting assessment - CALC:')
model = lgb.LGBMClassifier(min_child_samples=57)
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
y_pred = model.predict(x_test_calc)
cm = confusion_matrix(y_test_calc, y_pred)
print(cm)
print(classification_report(y_test_calc, y_pred, target_names=target_names_calc))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A0', 'A2', 'A3', 'A4', 'A5'])
disp.plot()
plt.show()

print('Predicting assessment - MASS:')
model = lgb.LGBMClassifier(max_depth=11)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
y_pred = model.predict(x_test_mass)
cm = confusion_matrix(y_test_mass, y_pred)
print(cm)
print(classification_report(y_test_mass, y_pred, target_names=target_names_mass))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
disp.plot()
plt.show()

# Target is pathology
x_train_calc = train_calc.drop(['pathology'], axis=1)
y_train_calc = train_calc['pathology']
x_test_calc = test_calc.drop(['pathology'], axis=1)
y_test_calc = test_calc['pathology']

x_train_mass = train_mass.drop(['pathology'], axis=1)
y_train_mass = train_mass['pathology']
x_test_mass = test_mass.drop(['pathology'], axis=1)
y_test_mass = test_mass['pathology']

print('Predicting pathology - CALC:')
model = lgb.LGBMClassifier(max_depth=5)
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
y_pred = model.predict(x_test_calc)
cm = confusion_matrix(y_test_calc, y_pred)
print(cm)
print(classification_report(y_test_calc, y_pred, target_names=target_names_path))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BWC', 'B', 'M'])
disp.plot()
plt.show()

print('Predicting pathology - MASS:')
model = lgb.LGBMClassifier(learning_rate=0.1, max_depth=5, min_child_samples=50, num_leaves=10)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
y_pred = model.predict(x_test_mass)
cm = confusion_matrix(y_test_mass, y_pred)
print(cm)
print(classification_report(y_test_mass, y_pred, target_names=target_names_path))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BWC', 'B', 'M'])
disp.plot()
plt.show()