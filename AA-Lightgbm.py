from Preprocessed_data2 import *
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

## Data preprocessing

y_train_calc = train_calc['overall BI-RADS assessment']
y_train_calc2 = train_calc['pathology']
x_train_calc = train_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_train_calc2 = train_calc.drop(['pathology'], axis=1)
y_test_calc = test_calc['overall BI-RADS assessment']
y_test_calc2 = test_calc['pathology']
x_test_calc = test_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_test_calc2 = test_calc.drop(['pathology'], axis=1)

y_train_mass = train_mass['overall BI-RADS assessment']
y_train_mass2 = train_mass['pathology']
x_train_mass = train_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_train_mass2 = train_mass.drop(['pathology'], axis=1)
y_test_mass = test_mass['overall BI-RADS assessment']
y_test_mass2 = test_mass['pathology']
x_test_mass = test_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_test_mass2 = test_mass.drop(['pathology'], axis=1)

## Feature selection

# Includes abnormality id but not left and right as features
print('Includes abnormality id but not left and right as features')
print('CALC')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
y_pred = model.predict(x_test_calc)
print(confusion_matrix(y_test_calc2, y_pred))

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc2, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc2,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc2,y_test_calc2)))

print('MASS')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
y_pred = model.predict(x_test_mass)
print(model.feature_importances_)
print(confusion_matrix(y_test_mass, y_pred))

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass2, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass2,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass2,y_test_mass2)))

# Does not include abnromality id or left and right as features

x_train_calc = x_train_calc.drop(['number of abnormalities'], axis=1)
x_train_calc2 = x_train_calc2.drop(['number of abnormalities'], axis=1)
x_test_calc = x_test_calc.drop(['number of abnormalities'], axis=1)
x_test_calc2 = x_test_calc2.drop(['number of abnormalities'], axis=1)
x_train_mass = x_train_mass.drop(['number of abnormalities'], axis=1)
x_train_mass2 = x_train_mass2.drop(['number of abnormalities'], axis=1)
x_test_mass = x_test_mass.drop(['number of abnormalities'], axis=1)
x_test_mass2 = x_test_mass2.drop(['number of abnormalities'], axis=1)

print('\nDoes not include abnromality id or left and right as features')
print('CALC')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
y_pred = model.predict(x_test_calc)

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc2, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc2,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc2,y_test_calc2)))

print('MASS')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
y_pred = model.predict(x_test_mass)

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass2, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass2,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass2,y_test_mass2)))

# Includes left and right but not abnromality id as a feature

x_train_calc = train_calc_2.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_test_calc = test_calc_2.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_train_calc2 = train_calc_2.drop(['pathology'], axis=1)
x_test_calc2 = test_calc_2.drop(['pathology'], axis=1)
x_train_mass = train_mass_2.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_test_mass = test_mass_2.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
x_train_mass2 = train_mass_2.drop(['pathology'], axis=1)
x_test_mass2 = test_mass_2.drop(['pathology'], axis=1)

print('\nIncludes left and right but not abnromality id as a feature')
print('CALC')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))
y_pred = model.predict(x_test_calc)

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_calc2, y_train_calc2)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc2,y_train_calc2)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc2,y_test_calc2)))

print('MASS')
print('Predicting assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))
y_pred = model.predict(x_test_mass)

print('Predicting pathology:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass2)))

print('Predicting pathology using assessment:')
model = lgb.LGBMClassifier()
model.fit(x_train_mass2, y_train_mass2)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass2,y_train_mass2)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass2,y_test_mass2)))

## Hypertuning conducted on training data including number of abnormalities but not left/right as feature

# Hyperparameter tuning for predicting assessment for calc

train_calc, validation_calc = train_test_split(train_calc, test_size=0.125, random_state=42)
y_train_calc = train_calc['overall BI-RADS assessment']
x_train_calc = train_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_validation_calc = validation_calc['overall BI-RADS assessment']
x_validation_calc = validation_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)

# Original accuracy
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print(model.score(x_validation_calc, y_validation_calc))

fig, axs = plt.subplots(3, 3)
fig.suptitle('Hyperparameter tuning for calc predicting assessment')

# Number of leaves in tree
accuracy = []
num_leaves = range(8, 200)
for num in num_leaves:
    model = lgb.LGBMClassifier(num_leaves=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0, 0].plot(num_leaves, accuracy)
axs[0, 0].set_xlabel('Number of leaves')
accuracy = np.array(accuracy)
print(f'Maximum train accuracy for validation calc is: {np.max(accuracy)}')
print(f'Number of leaves for maximum accuracy is: {num_leaves[np.argmax(accuracy)]}')
max_leaves = num_leaves[np.argmax(accuracy)]

# Max depth
accuracy = []
values = range(3,16)
depth = list(values)
depth.append(-1)
for d in depth:
    model = lgb.LGBMClassifier(max_depth=d)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0,1].plot(depth, accuracy)
axs[0,1].set_xlabel('Maximum depth')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Max depth for maximum accuracy is: {depth[np.argmax(accuracy)]}')
max_d = depth[np.argmax(accuracy)]

# Minimum data in leaf
accuracy = []
leaf_data = range(5,150)
for data in leaf_data:
    model = lgb.LGBMClassifier(min_child_samples=data)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0,2].plot(leaf_data, accuracy)
axs[0,2].set_xlabel('Minimum number of data in leaf')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Minimum data in leaf for maximum accuracy is: {leaf_data[np.argmax(accuracy)]}')
min_leaf = leaf_data[np.argmax(accuracy)]

# Minimum gain to split
accuracy = []
min_gain = []
for i in range (0, 100):
    gain = i/10
    min_gain.append(gain)
    model = lgb.LGBMClassifier(min_split_gain=gain)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1,0].plot(min_gain, accuracy)
axs[1,0].set_xlabel('Minimum split gain')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Minimum split gain for maximum accuracy is: {min_gain[np.argmax(accuracy)]}')
min_split = min_gain[np.argmax(accuracy)]

# Learning rate
accuracy = []
learning_rate = []
for i in range (1, 50):
    rate = i/100
    learning_rate.append(rate)
    model = lgb.LGBMClassifier(learning_rate=rate)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1, 1].plot(learning_rate, accuracy)
axs[1, 1].set_xlabel('Learning rate')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Learning rate for maximum accuracy is: {learning_rate[np.argmax(accuracy)]}')
max_rate = learning_rate[np.argmax(accuracy)]

# Number of estimators
accuracy = []
num_iterations = range(75,125)
for num in num_iterations:
    model = lgb.LGBMClassifier(n_estimators=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1,2].plot(num_iterations, accuracy)
axs[1,2].set_xlabel('Number of iterations')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Number of estimators for maximum accuracy is: {num_iterations[np.argmax(accuracy)]}')
max_estimators = num_iterations[np.argmax(accuracy)]

# Max bin
accuracy = []
num_bin = range(100, 300, 5)
for num in num_bin:
    model = lgb.LGBMClassifier(max_bin=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2,0].plot(num_bin, accuracy)
axs[2,0].set_xlabel('Max bin size')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Maximum bin size for maximum accuracy is: {num_bin[np.argmax(accuracy)]}')
max_bin_size = num_bin[np.argmax(accuracy)]

# Bagging Fraction
accuracy = []
fractions = []
for i in range (5, 11):
    frac = i/10
    fractions.append(frac)
    model = lgb.LGBMClassifier(subsample=frac)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2, 1].plot(fractions, accuracy)
axs[2, 1].set_xlabel('Bagging fraction')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {fractions[np.argmax(accuracy)]}')
max_frac = fractions[np.argmax(accuracy)]

# Bagging Frequency
accuracy = []
frequencies = range(0, 10)
for freq in frequencies:
    model = lgb.LGBMClassifier(subsample_freq=freq)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2, 2].plot(frequencies, accuracy)
axs[2, 2].set_xlabel('Bagging frequency')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {frequencies[np.argmax(accuracy)]}')
max_freq = frequencies[np.argmax(accuracy)]

for ax in axs.flat:
    ax.set(ylabel='Accuracy')
fig.tight_layout()
plt.show()

# Hyperparamter turning for Mass predicting assessment

train_mass, validation_mass = train_test_split(train_mass, test_size=0.125, random_state=42)
y_train_mass = train_mass['overall BI-RADS assessment']
x_train_mass = train_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_validation_mass = validation_mass['overall BI-RADS assessment']
x_validation_mass = validation_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)

# Original accuracy
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print(model.score(x_validation_mass, y_validation_mass))

fig, axs = plt.subplots(3, 3)
fig.suptitle('Hyperparameter tuning for mass predicting assessment')

# Number of leaves in tree
accuracy = []
num_leaves = range(8, 200)
for num in num_leaves:
    model = lgb.LGBMClassifier(num_leaves=num)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0, 0].plot(num_leaves, accuracy)
axs[0, 0].set_xlabel('Number of leaves')
accuracy = np.array(accuracy)
print(f'Maximum train accuracy for validation mass is: {np.max(accuracy)}')
print(f'Number of leaves for maximum accuracy is: {num_leaves[np.argmax(accuracy)]}')
max_leaves = num_leaves[np.argmax(accuracy)]

# Max depth
accuracy = []
values = range(3,16)
depth = list(values)
depth.append(-1)
for d in depth:
    model = lgb.LGBMClassifier(max_depth=d)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0,1].plot(depth, accuracy)
axs[0,1].set_xlabel('Maximum depth')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Max depth for maximum accuracy is: {depth[np.argmax(accuracy)]}')
max_d = depth[np.argmax(accuracy)]

# Minimum data in leaf
accuracy = []
leaf_data = range(5,150)
for data in leaf_data:
    model = lgb.LGBMClassifier(min_child_samples=data)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0,2].plot(leaf_data, accuracy)
axs[0,2].set_xlabel('Minimum number of data in leaf')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Minimum data in leaf for maximum accuracy is: {leaf_data[np.argmax(accuracy)]}')
min_leaf = leaf_data[np.argmax(accuracy)]

# Minimum gain to split
accuracy = []
min_gain = []
for i in range (0, 100):
    gain = i/10
    min_gain.append(gain)
    model = lgb.LGBMClassifier(min_split_gain=gain, min_child_samples=min_leaf)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1,0].plot(min_gain, accuracy)
axs[1,0].set_xlabel('Minimum split gain')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Minimum split gain for maximum accuracy is: {min_gain[np.argmax(accuracy)]}')
min_split = min_gain[np.argmax(accuracy)]

# Learning rate
accuracy = []
learning_rate = []
for i in range (1, 50):
    rate = i/100
    learning_rate.append(rate)
    model = lgb.LGBMClassifier(learning_rate=rate, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1, 1].plot(learning_rate, accuracy)
axs[1, 1].set_xlabel('Learning rate')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Learning rate for maximum accuracy is: {learning_rate[np.argmax(accuracy)]}')
max_rate = learning_rate[np.argmax(accuracy)]

# Number of estimators
accuracy = []
num_iterations = range(75,125)
for num in num_iterations:
    model = lgb.LGBMClassifier(n_estimators=num, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1,2].plot(num_iterations, accuracy)
axs[1,2].set_xlabel('Number of iterations')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Number of estimators for maximum accuracy is: {num_iterations[np.argmax(accuracy)]}')
max_estimators = num_iterations[np.argmax(accuracy)]

# Max bin
accuracy = []
num_bin = range(100, 300, 5)
for num in num_bin:
    model = lgb.LGBMClassifier(max_bin=num, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2,0].plot(num_bin, accuracy)
axs[2,0].set_xlabel('Max bin size')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Maximum bin size for maximum accuracy is: {num_bin[np.argmax(accuracy)]}')
max_bin_size = num_bin[np.argmax(accuracy)]

# Bagging Fraction
accuracy = []
fractions = []
for i in range (5, 11):
    frac = i/10
    fractions.append(frac)
    model = lgb.LGBMClassifier(min_child_samples=45, subsample=frac)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2, 1].plot(fractions, accuracy)
axs[2, 1].set_xlabel('Bagging fraction')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {fractions[np.argmax(accuracy)]}')
max_frac = fractions[np.argmax(accuracy)]

# Bagging Frequency
accuracy = []
frequencies = range(0, 10)
for freq in frequencies:
    model = lgb.LGBMClassifier(min_child_samples=45, subsample_freq=freq)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2, 2].plot(frequencies, accuracy)
axs[2, 2].set_xlabel('Bagging frequency')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {frequencies[np.argmax(accuracy)]}')
max_freq = frequencies[np.argmax(accuracy)]

for ax in axs.flat:
    ax.set(ylabel='Accuracy')
fig.tight_layout()
plt.show()

# Hyperparameter tuning for calc predicting pathology 

train_calc, validation_calc = train_test_split(train_calc, test_size=0.125, random_state=42)
y_train_calc = train_calc['pathology']
x_train_calc = train_calc.drop(['pathology'], axis=1)
y_validation_calc = validation_calc['pathology']
x_validation_calc = validation_calc.drop(['pathology'], axis=1)

# Original accuracy
model = lgb.LGBMClassifier()
model.fit(x_train_calc, y_train_calc)
print(model.score(x_validation_calc, y_validation_calc))

fig, axs = plt.subplots(3, 3)
fig.suptitle('Hyperparameter tuning for calc predicting pathology')

# Number of leaves in tree
accuracy = []
num_leaves = range(8, 200)
for num in num_leaves:
    model = lgb.LGBMClassifier(num_leaves=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0, 0].plot(num_leaves, accuracy)
axs[0, 0].set_xlabel('Number of leaves')
accuracy = np.array(accuracy)
print(f'Maximum train accuracy for validation calc is: {np.max(accuracy)}')
print(f'Number of leaves for maximum accuracy is: {num_leaves[np.argmax(accuracy)]}')
max_leaves = num_leaves[np.argmax(accuracy)]

# Max depth
accuracy = []
values = range(3,16)
depth = list(values)
depth.append(-1)
for d in depth:
    model = lgb.LGBMClassifier(max_depth=d)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0,1].plot(depth, accuracy)
axs[0,1].set_xlabel('Maximum depth')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Max depth for maximum accuracy is: {depth[np.argmax(accuracy)]}')
max_d = depth[np.argmax(accuracy)]

# Minimum data in leaf
accuracy = []
leaf_data = range(5,150)
for data in leaf_data:
    model = lgb.LGBMClassifier(min_child_samples=data)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[0,2].plot(leaf_data, accuracy)
axs[0,2].set_xlabel('Minimum number of data in leaf')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Minimum data in leaf for maximum accuracy is: {leaf_data[np.argmax(accuracy)]}')
min_leaf = leaf_data[np.argmax(accuracy)]

# Minimum gain to split
accuracy = []
min_gain = []
for i in range (0, 100):
    gain = i/10
    min_gain.append(gain)
    model = lgb.LGBMClassifier(min_split_gain=gain)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1,0].plot(min_gain, accuracy)
axs[1,0].set_xlabel('Minimum split gain')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Minimum split gain for maximum accuracy is: {min_gain[np.argmax(accuracy)]}')
min_split = min_gain[np.argmax(accuracy)]

# Learning rate
accuracy = []
learning_rate = []
for i in range (1, 50):
    rate = i/100
    learning_rate.append(rate)
    model = lgb.LGBMClassifier(learning_rate=rate)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1, 1].plot(learning_rate, accuracy)
axs[1, 1].set_xlabel('Learning rate')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Learning rate for maximum accuracy is: {learning_rate[np.argmax(accuracy)]}')
max_rate = learning_rate[np.argmax(accuracy)]

# Number of estimators
accuracy = []
num_iterations = range(75,125)
for num in num_iterations:
    model = lgb.LGBMClassifier(n_estimators=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[1,2].plot(num_iterations, accuracy)
axs[1,2].set_xlabel('Number of iterations')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Number of estimators for maximum accuracy is: {num_iterations[np.argmax(accuracy)]}')
max_estimators = num_iterations[np.argmax(accuracy)]

# Max bin
accuracy = []
num_bin = range(100, 300, 5)
for num in num_bin:
    model = lgb.LGBMClassifier(max_bin=num)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2,0].plot(num_bin, accuracy)
axs[2,0].set_xlabel('Max bin size')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Maximum bin size for maximum accuracy is: {num_bin[np.argmax(accuracy)]}')
max_bin_size = num_bin[np.argmax(accuracy)]

# Bagging Fraction
accuracy = []
fractions = []
for i in range (5, 11):
    frac = i/10
    fractions.append(frac)
    model = lgb.LGBMClassifier(subsample=frac)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2, 1].plot(fractions, accuracy)
axs[2, 1].set_xlabel('Bagging fraction')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {fractions[np.argmax(accuracy)]}')
max_frac = fractions[np.argmax(accuracy)]

# Bagging Frequency
accuracy = []
frequencies = range(0, 10)
for freq in frequencies:
    model = lgb.LGBMClassifier(subsample_freq=freq)
    model.fit(x_train_calc, y_train_calc)
    acc = model.score(x_validation_calc, y_validation_calc)
    accuracy.append(acc)
axs[2, 2].plot(frequencies, accuracy)
axs[2, 2].set_xlabel('Bagging frequency')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation calc is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {frequencies[np.argmax(accuracy)]}')
max_freq = frequencies[np.argmax(accuracy)]

for ax in axs.flat:
    ax.set(ylabel='Accuracy')
fig.tight_layout()
plt.show()

# Hyperparamter turning for Mass predicting pathology

train_mass, validation_mass = train_test_split(train_mass, test_size=0.125, random_state=42)
y_train_mass = train_mass['pathology']
x_train_mass = train_mass.drop(['pathology'], axis=1)
y_validation_mass = validation_mass['pathology']
x_validation_mass = validation_mass.drop(['pathology'], axis=1)

# Original accuracy
model = lgb.LGBMClassifier()
model.fit(x_train_mass, y_train_mass)
print(model.score(x_validation_mass, y_validation_mass))

fig, axs = plt.subplots(3, 3)
fig.suptitle('Hyperparameter tuning for mass')

# Number of leaves in tree
accuracy = []
num_leaves = range(8, 200)
for num in num_leaves:
    model = lgb.LGBMClassifier(num_leaves=num)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0, 0].plot(num_leaves, accuracy)
axs[0, 0].set_xlabel('Number of leaves')
accuracy = np.array(accuracy)
print(f'Maximum train accuracy for validation mass is: {np.max(accuracy)}')
print(f'Number of leaves for maximum accuracy is: {num_leaves[np.argmax(accuracy)]}')
max_leaves = num_leaves[np.argmax(accuracy)]

# Max depth
accuracy = []
values = range(3,16)
depth = list(values)
depth.append(-1)
for d in depth:
    model = lgb.LGBMClassifier(max_depth=d)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0,1].plot(depth, accuracy)
axs[0,1].set_xlabel('Maximum depth')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Max depth for maximum accuracy is: {depth[np.argmax(accuracy)]}')
max_d = depth[np.argmax(accuracy)]

# Minimum data in leaf
accuracy = []
leaf_data = range(5,150)
for data in leaf_data:
    model = lgb.LGBMClassifier(min_child_samples=data)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[0,2].plot(leaf_data, accuracy)
axs[0,2].set_xlabel('Minimum number of data in leaf')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Minimum data in leaf for maximum accuracy is: {leaf_data[np.argmax(accuracy)]}')
min_leaf = leaf_data[np.argmax(accuracy)]

# Minimum gain to split
accuracy = []
min_gain = []
for i in range (0, 100):
    gain = i/10
    min_gain.append(gain)
    model = lgb.LGBMClassifier(min_split_gain=gain, min_child_samples=min_leaf)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1,0].plot(min_gain, accuracy)
axs[1,0].set_xlabel('Minimum split gain')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Minimum split gain for maximum accuracy is: {min_gain[np.argmax(accuracy)]}')
min_split = min_gain[np.argmax(accuracy)]

# Learning rate
accuracy = []
learning_rate = []
for i in range (1, 50):
    rate = i/100
    learning_rate.append(rate)
    model = lgb.LGBMClassifier(learning_rate=rate, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1, 1].plot(learning_rate, accuracy)
axs[1, 1].set_xlabel('Learning rate')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Learning rate for maximum accuracy is: {learning_rate[np.argmax(accuracy)]}')
max_rate = learning_rate[np.argmax(accuracy)]

# Number of estimators
accuracy = []
num_iterations = range(75,125)
for num in num_iterations:
    model = lgb.LGBMClassifier(n_estimators=num, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[1,2].plot(num_iterations, accuracy)
axs[1,2].set_xlabel('Number of iterations')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Number of estimators for maximum accuracy is: {num_iterations[np.argmax(accuracy)]}')
max_estimators = num_iterations[np.argmax(accuracy)]

# Max bin
accuracy = []
num_bin = range(100, 300, 5)
for num in num_bin:
    model = lgb.LGBMClassifier(max_bin=num, min_child_samples=45)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2,0].plot(num_bin, accuracy)
axs[2,0].set_xlabel('Max bin size')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Maximum bin size for maximum accuracy is: {num_bin[np.argmax(accuracy)]}')
max_bin_size = num_bin[np.argmax(accuracy)]

# Bagging Fraction
accuracy = []
fractions = []
for i in range (5, 11):
    frac = i/10
    fractions.append(frac)
    model = lgb.LGBMClassifier(min_child_samples=45, subsample=frac)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2, 1].plot(fractions, accuracy)
axs[2, 1].set_xlabel('Bagging fraction')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {fractions[np.argmax(accuracy)]}')
max_frac = fractions[np.argmax(accuracy)]

# Bagging Frequency
accuracy = []
frequencies = range(0, 10)
for freq in frequencies:
    model = lgb.LGBMClassifier(min_child_samples=45, subsample_freq=freq)
    model.fit(x_train_mass, y_train_mass)
    acc = model.score(x_validation_mass, y_validation_mass)
    accuracy.append(acc)
axs[2, 2].plot(frequencies, accuracy)
axs[2, 2].set_xlabel('Bagging frequency')
accuracy = np.array(accuracy)
print(f'Maximum accuracy for validation mass is: {np.max(accuracy)}')
print(f'Bagging fraction for maximum accuracy is: {frequencies[np.argmax(accuracy)]}')
max_freq = frequencies[np.argmax(accuracy)]

for ax in axs.flat:
    ax.set(ylabel='Accuracy')
fig.tight_layout()
plt.show()

# Final models produced using iterative hyperparameter tuning
print('FINAL MODELS:')
x_train_calc = train_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_train_calc = train_calc['overall BI-RADS assessment']
x_test_calc = test_calc.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_calc = test_calc['overall BI-RADS assessment']

x_train_mass = train_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_train_mass = train_mass['overall BI-RADS assessment']
x_test_mass = test_mass.drop(['overall BI-RADS assessment', 'pathology'], axis=1)
y_test_mass = test_mass['overall BI-RADS assessment']

print('Predicting assessment - CALC:')
model = lgb.LGBMClassifier(min_child_samples=57)
model.fit(x_train_calc, y_train_calc)
print('\tTraining calc accuracy {:.4f}'.format(model.score(x_train_calc,y_train_calc)))
print('\tTesting calc accuracy {:.4f}'.format(model.score(x_test_calc,y_test_calc)))

print('Predicting assessment - MASS:')
model = lgb.LGBMClassifier(max_depth=11)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))

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

print('Predicting pathology - MASS:')
model = lgb.LGBMClassifier(min_child_samples=5)
model.fit(x_train_mass, y_train_mass)
print('\tTraining mass accuracy {:.4f}'.format(model.score(x_train_mass,y_train_mass)))
print('\tTesting mass accuracy {:.4f}'.format(model.score(x_test_mass,y_test_mass)))