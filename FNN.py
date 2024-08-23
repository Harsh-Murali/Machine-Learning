# The result of model training would be different in each execution.
# Some part of tuning has been removed for the sake of simplicity.

# Import the libraries
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # to avoid CUDA error
import math
import copy
import itertools
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Initial Settings
batch_size = 64
hidden_size = 512
learning_rate = 1e-3
epochs = 10

iteration = 50 # number of iterations for each epoch

# Import the dataset
train_calc = pd.read_csv('calc_case_description_train_set.csv')
test_calc = pd.read_csv('calc_case_description_test_set.csv')
train_mass = pd.read_csv('mass_case_description_train_set.csv')
test_mass = pd.read_csv('mass_case_description_test_set.csv')

# Merge
calc = train_calc.values.tolist() + test_calc.values.tolist()
calc = pd.DataFrame(calc, columns = train_calc.columns)
mass = train_mass.values.tolist() + test_mass.values.tolist()
mass = pd.DataFrame(mass, columns = train_mass.columns)

# embedding sizes
calc_type_embedding_size = 0
calc_dist_embedding_size = 0
mass_shape_embedding_size = 0
mass_margins_embedding_size = 0

# Function to create dictionaries for categorical features
def create_embedding_dict(series):
    unique_values = series.unique()
    embedding_dict = {value: i for i, value in enumerate(unique_values)}
    return embedding_dict

# Function to process the data
def process(data):
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
        BENIGN = 1
        MALIGNANT = 2
        '''
        data['pathology'] = data['pathology'].map({'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 1, 'MALIGNANT': 2})
        
        # Embedding sizes will be saved as global variables
        global calc_type_embedding_size
        global calc_dist_embedding_size
        global mass_shape_embedding_size
        global mass_margins_embedding_size
        
        # Create embedding dictionaries for categorical features then
        # Define embedding sizes
        try:
                calc_type_embedding_dict = create_embedding_dict(data['calc type'])
                calc_dist_embedding_dict = create_embedding_dict(data['calc distribution'])
                calc_type_embedding_size = len(calc_type_embedding_dict)
                calc_dist_embedding_size = len(calc_dist_embedding_dict)
        except KeyError:
                mass_shape_embedding_dict = create_embedding_dict(data['mass shape'])
                mass_margins_embedding_dict = create_embedding_dict(data['mass margins'])
                mass_shape_embedding_size = len(mass_shape_embedding_dict)
                mass_margins_embedding_size = len(mass_margins_embedding_dict)
        
        # Replace categorical values with their embedding indices        
        try:
                data['calc type'] = data['calc type'].map(calc_type_embedding_dict)
                data['calc distribution'] = data['calc distribution'].map(calc_dist_embedding_dict)
        except KeyError:
                data['mass shape'] = data['mass shape'].map(mass_shape_embedding_dict)
                data['mass margins'] = data['mass margins'].map(mass_margins_embedding_dict)
        
        # Rename columns
        data.rename(columns={'abnormality id': 'number of abnormalities', 
                             'assessment' : 'overall BI-RADS assessment'}, inplace=True)
        try:
                data.rename(columns={'breast_density' : 'breast density'}, inplace=True)
        except KeyError:
                pass
        return data
    
# Data Processing
calc = process(calc)
mass = process(mass)
# Split the dataset into train and test sets
train_calc, test_calc = calc[:train_calc.shape[0]], calc[train_calc.shape[0]:]
train_mass, test_mass = mass[:train_mass.shape[0]], mass[train_mass.shape[0]:]

# label maps (will not be used to train or evaluate the model)
labels_map_assessment = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5"
}

labels_map_pathology = {
    0: "BENIGN_WITHOUT_CALLBACK",
    1: "BENIGN",
    2: "MALIGNANT"
}

# Class to create the Dataset
class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Class to get the datasets and dataloaders for training and testing sets
def get_dataloaders(train, test, type, batch_size):
    if type == "a":
        # calc, assessement prediction
        X_train = train.drop(['overall BI-RADS assessment', 'pathology'], axis=1).values
        y_train = train['overall BI-RADS assessment'].values
        X_test = test.drop(['overall BI-RADS assessment', 'pathology'], axis=1).values
        y_test = test['overall BI-RADS assessment'].values
    elif type == "p":
        # calc, pathology prediction
        X_train = train.drop('pathology', axis=1).values
        y_train = train['pathology'].values
        X_test = test.drop('pathology', axis=1).values
        y_test = test['pathology'].values
        
    # Assuming you have already preprocessed the data and split it into training and testing sets    
    train_dataset = CreateDataset(X_train, y_train)
    test_dataset = CreateDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataset, test_dataset, train_loader, test_loader

# Using cuba is possible
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

'''
:param: input_size: number of features
:param: hidden_size: number of neurons in the hidden layer
:param: num_classes: number of output classes
:param: hidden_layers: number of hidden layers
:param: activation: activation function
:param: weight_init: weight initialization method
'''
class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, hidden_layers=1, activation="relu", weight_init="lecun"):
        super().__init__()
        # Stacks to save the functions
        self.linear_non_linear_stack = nn.Sequential().to(device)
        
        # Save them into the list first
        modules = []
        # Input Layer
        modules.append(nn.Linear(input_size, hidden_size))
        
        # weight initialization
        if weight_init == "lecun":
            pass
        elif weight_init == "zero":
            nn.init.zeros_(modules[0].weight)
        elif weight_init == "normal":
            nn.init.normal_(modules[0].weight, mean=0, std=0.01)
        elif weight_init == "xavier" or weight_init == "glorot":
            nn.init.xavier_normal_(modules[0].weight)
        elif weight_init == "kaiming" or weight_init == "he":
            nn.init.kaiming_normal_(modules[0].weight)
        
        # activation function
        if activation == "relu":
            modules.append(nn.ReLU())
        elif activation == "sigmoid":
            modules.append(nn.Sigmoid())
        elif activation == "tanh":
            modules.append(nn.Tanh())
        elif activation == "leaky_relu":
            modules.append(nn.LeakyReLU())
        
        # Hidden Layers
        for i in range(hidden_layers):
            ln = nn.Linear(hidden_size, hidden_size)
            
            if weight_init == "lecun":
                pass
            elif weight_init == "zero":
                nn.init.zeros_(ln.weight)
            elif weight_init == "normal":
                nn.init.normal_(ln.weight, mean=0, std=0.01)
            elif weight_init == "xavier" or weight_init == "glorot":
                nn.init.xavier_normal_(ln.weight)
            elif weight_init == "kaiming" or weight_init == "he":
                nn.init.kaiming_normal_(ln.weight)
            
            modules.append(ln)
            if activation == "relu":
                modules.append(nn.ReLU())
            elif activation == "sigmoid":
                modules.append(nn.Sigmoid())
            elif activation == "tanh":
                modules.append(nn.Tanh())
            elif activation == "leaky_relu":
                modules.append(nn.LeakyReLU())
        
        # output layer
        modules.append(nn.Linear(hidden_size, num_classes))
        
        # put the list into the stack
        self.linear_non_linear_stack = nn.Sequential(*modules).to(device)

    def forward(self, x):
        x = x.to(device)
        logits = self.linear_non_linear_stack(x)
        return logits
    
# train model
def train_loop(dataloader, model, loss_fn, optimizer, p="False", scheduler=None):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()  # Move input and target tensors to the same device as the model
        # Compute prediction and loss
        pred = model(X).to(device)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # will be print only when asked - I didn't use it but just in case
        if batch % 10 == 0 or batch == len(dataloader) - 1:
            loss, current = loss.item(), (batch + 1) * len(X)
            if batch == len(dataloader) - 1:
                current = size
            if p == True:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# evaluate model
def test_loop(dataloader, model, loss_fn, p="False"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).long()  # Move input and target tensors to the same device as the model
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # argmax returns the index of the max value

    test_loss /= num_batches
    correct /= size
    # will be print only when asked - I didn't use it but just in case
    if p == True:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return round(100*correct, 1), test_loss # accuracy, loss

# Types of optimizers
optimizer_types = ["minibGD", "SGDMomentum", "SGDNestrov", "Adam", "Adagrad", "Adadelta", "RMSprop"]

# Set of activation functions
activations = ["relu", "sigmoid", "tanh", "leaky_relu"]

# Set of weight initializations
weight_inits = ["lecun", "zero", "normal", "xavier", "kaiming"]

# Set of hidden layers
hidden_layers_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Set of number of neurons in the hidden layers
hidden_sizes = [32, 64, 128, 256, 512, 1024, 2048]

# Set of learning rates 
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# Set of batch sizes
batch_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]

# Set of epochs
possible_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Regularization
regularizations = ["L1", "L2", "Dropout", "BatchNorm", "None"]
epochs_candidates = [10, 20, 30]
l1_lambda_candidates = [0.001, 0.01, 0.1, 1, 10, 100]
l2_lambda_candidates = [0.001, 0.01, 0.1, 1, 10, 100]
dropout_candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
batchnorm_candidates = [True, False]

# Create l1 regularization 
def l1_regularizer(model, l1_lambda):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return l1_lambda * l1_loss

# Function to train the model with l1 regularization
def train_L1(dataloader, model, loss_fn, optimizer, l1_lambda, p="False", scheduler=None):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device).long()  # Move input and target tensors to the same device as the model
        # Compute prediction and loss
        pred = model(X).to(device)
        
        loss = loss_fn(pred, y)

        # Compute L1 regularization penalty
        l1_penalty = l1_regularizer(model, l1_lambda)

        # Combine original loss and L1 penalty
        loss = loss + l1_penalty
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0 or batch == len(dataloader) - 1:
            loss, current = loss.item(), (batch + 1) * len(X)
            if batch == len(dataloader) - 1:
                current = size
            if p == True:
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# optimzer function
def get_optimizer(model, type, lr, weight_decay=0):
    if type == "minibGD": # mini batch gradient descent
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0)
    elif type == "SGDMomentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    elif type == "SGDNestrov":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
    elif type == "Adam":
        return torch.optim.Adam(model.parameters(), weight_decay=0)
    # elif type == "Adamax":
        # return torch.optim.Adamax(model.parameters())
    elif type == "Adagrad":
        return torch.optim.Adagrad(model.parameters(), weight_decay=0)
    elif type == "Adadelta":
        return torch.optim.Adadelta(model.parameters(), weight_decay=0)
    elif type == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), weight_decay=0)
# Adamax has been removed
'''
Adamax, a variant of Adam based on the infinity norm, is a first-order gradient-based 
optimization method. Due to its capability of adjusting the learning rate based on data 
characteristics, it is suited to learn time-variant process, e.g., speech data with 
dynamically changed noise conditions. - keras.io
'''

# Model and its data will be saved on this class
class Model:
    def __init__(self, epochs, batch_size, train, test, Dtype, hidden_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.train_dataset, self.test_dataset, self.train_dataloader, self.test_dataloader = get_dataloaders(train, test, Dtype, batch_size)

        self.input_size = self.train_dataloader.dataset.X.shape[1]
        self.hidden_size = hidden_size
        self.num_classes = 6 if Dtype == "a" else 3
        
        self.lr = learning_rate
        self.accuracy = 0
        
        self.optimizer_type = "minibGD"
        self.weight_init = "lecun"
        self.actv = "relu"
        self.epochs = epochs
        self.hidden_layers = 1
        
    def get_model(self, device, optimizer_type=None, lr=None, hidden_size=None):
        if optimizer_type == None:
            optimizer_type = self.optimizer_type
        if lr == None:
            lr = self.lr
        if hidden_size == None:
            hidden_size = self.hidden_size
            
        model = FNN(self.input_size, hidden_size, self.num_classes).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = get_optimizer(model, optimizer_type, lr)
        return model, loss_fn, optimizer
        
    def train_and_evaluate(self, train_dataloader, test_dataloader, model, loss_fn, optimizer):
        for t in range(self.epochs):
            train_loop(train_dataloader, model, loss_fn, optimizer)
            accuracy = test_loop(test_dataloader, model, loss_fn)[0]
        # update when meet better accuracy
        if accuracy > self.accuracy:
            self.accuracy = accuracy
            self.model = model
            self.loss_fn = loss_fn
            self.optimizer = optimizer
            return accuracy
        return None

# Features -> Assessment (Calcification)
fac = Model(epochs, batch_size, train_calc, test_calc, "a", hidden_size, learning_rate)
model, loss_fn, optimizer = fac.get_model(device)
fac.train_and_evaluate(fac.train_dataloader, fac.test_dataloader, model, loss_fn, optimizer)
# Features -> Assessment (Mass)
fam = Model(epochs, batch_size, train_mass, test_mass, "a", hidden_size, learning_rate)
model, loss_fn, optimizer = fam.get_model(device)
fam.train_and_evaluate(fam.train_dataloader, fam.test_dataloader, model, loss_fn, optimizer)
# Features including Assessment -> Pathology (Calcification)
fpc = Model(epochs, batch_size, train_calc, test_calc, "p", hidden_size, learning_rate)
model, loss_fn, optimizer = fpc.get_model(device)
fpc.train_and_evaluate(fpc.train_dataloader, fpc.test_dataloader, model, loss_fn, optimizer)
# Features including Assessment -> Pathology (Mass)
fpm = Model(epochs, batch_size, train_mass, test_mass, "p", hidden_size, learning_rate)
model, loss_fn, optimizer = fpm.get_model(device)
fpm.train_and_evaluate(fpm.train_dataloader, fpm.test_dataloader, model, loss_fn, optimizer)

Models = [fac, fam, fpc, fpm]

# Find the best optimizer
def tune_opt(M):
    for optimizer_type in optimizer_types:
        print("Optimizer: ", optimizer_type)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, optimizer_type, M.lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best optimizer: ", optimizer_type)
                print("New best model found with accuracy: ", accuracy)
                M.optimizer_type = optimizer_type
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer
                
# Find the best learning rate
def tune_lr(M):
    for lr in learning_rates:
        print("Learning rate: ", lr)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best learning rate: ", lr)
                print("New best model found with accuracy: ", accuracy)
                M.lr = lr
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

# Find the best Activation function
def tune_actv(M):
    for actv in activations:
        print("Activation function: ", actv)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes, activation=actv).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best activation function: ", actv)
                print("New best model found with accuracy: ", accuracy)
                M.actv = actv
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

# Find the best weight initialization
def tune_weight_init(M):
    for weight_init in weight_inits:
        print("Weight initialization: ", weight_init)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes, activation=M.actv, weight_init=weight_init).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best weight initialization: ", weight_init)
                print("New best model found with accuracy: ", accuracy)
                M.weight_init = weight_init
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

# Find the best batch size
def tune_batch(M):
    for batch_size in batch_sizes:
        print("Batch size: ", batch_size)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes, activation=M.actv, weight_init=M.weight_init).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)
            train_dataloader = DataLoader(dataset=M.train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(dataset=M.test_dataset, batch_size=batch_size, shuffle=True)

            for t in range(M.epochs):
                train_loop(train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best batch size: ", batch_size)
                print("New best model found with accuracy: ", accuracy)
                M.batch_size = batch_size
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer
                M.train_dataloader = train_dataloader
                M.test_dataloader = test_dataloader

# Find the best number of neurons
def tune_hidden_size(M):
    for hidden_size in hidden_sizes:
        print("Number of neurons: ", hidden_size)
        for i in range(iteration):
            model = FNN(M.input_size, hidden_size, M.num_classes, activation=M.actv, weight_init=M.weight_init).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best number of neurons: ", hidden_size)
                print("New best model found with accuracy: ", accuracy)
                M.hidden_size = hidden_size
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

# find the best layers
def tune_hidden_layers(M):
    for hidden_layers in hidden_layers_list:
        print("Number of layers: ", hidden_layers)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes, activation=M.actv, weight_init=M.weight_init, hidden_layers=hidden_layers).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)

            for t in range(M.epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best number of layers: ", hidden_layers)
                print("New best model found with accuracy: ", accuracy)
                M.hidden_layers = hidden_layers
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

# find the best number of epochs
def tune_epochs(M):
    for epochs in epochs_candidates:
        print("Testing epochs: ", epochs)
        for i in range(iteration):
            model = FNN(M.input_size, M.hidden_size, M.num_classes, activation=M.actv, weight_init=M.weight_init, hidden_layers=M.hidden_layers).to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = get_optimizer(model, M.optimizer_type, M.lr)

            for t in range(epochs):
                train_loop(M.train_dataloader, model, loss_fn, optimizer)
                accuracy = test_loop(M.test_dataloader, model, loss_fn)[0]
            
            if accuracy > M.accuracy:
                print("New best number of epochs: ", epochs)
                print("New best model found with accuracy: ", accuracy)
                M.epochs = epochs
                M.accuracy = accuracy
                M.model = model
                M.loss_fn = loss_fn
                M.optimizer = optimizer

if __name__ == "__main__":
    model_name = ["Features to Assessment (calcification)", "Features to Assessment (mass)", 
                  "Features including Assessment to Pathology (calcification)",
                  "Features including Assessment to Pathology (mass)"]
    names = ["F2A_calc", "F2A_mass", "F2P_calc", "F2P_mass"]
    
    iteration = input("How many iteration for each parameter tuning? (default 50) ")
    try:
        iteration = int(iteration)
    except TypeError:
        iteration = 50
    
    for M in Models:
        print("Tuning model: ", model_name[Models.index(M)])
        tune_opt(M)
        tune_lr(M)
        tune_actv(M)
        tune_weight_init(M)
        tune_batch(M)
        tune_hidden_size(M)
        tune_hidden_layers(M)
        tune_epochs(M)
        print("\n Best model found with accuracy: ", M.accuracy)
        torch.save(M.model.state_dict(), "FNN_model_" + M.name + ".pt")
        print("Model saved as FNN_model_" + M.name + ".pt")