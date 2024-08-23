import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Import the dataset
train_calc = pd.read_csv('calc_case_description_train_set.csv')
test_calc = pd.read_csv('calc_case_description_test_set.csv')
train_mass = pd.read_csv('mass_case_description_train_set.csv')
test_mass = pd.read_csv('mass_case_description_test_set.csv')

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
        # '''
        # pathology :
        # BENIGN_WITHOUT_CALLBACK = 0
        # BENIGN = 1
        # MALIGNANT = 2
        # '''
        data['pathology'] = data['pathology'].map({'BENIGN_WITHOUT_CALLBACK': 0, 'BENIGN': 1, 'MALIGNANT': 2})
        
        # Create embedding dictionaries for categorical features
        # and define embedding sizes
        
        global calc_type_embedding_size
        global calc_dist_embedding_size
        global mass_shape_embedding_size
        global mass_margins_embedding_size
        
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
        
        # rename columns
        data.rename(columns={'abnormality id': 'number of abnormalities', 
                             'assessment' : 'overall BI-RADS assessment'}, inplace=True)
        try:
                data.rename(columns={'breast_density' : 'breast density'}, inplace=True)
                # split
        except KeyError:
                return data
        
        return data

# Execute the data process function
calc = process(calc)
mass = process(mass)
train_calc, test_calc = calc[:train_calc.shape[0]], calc[train_calc.shape[0]:]
train_mass, test_mass = mass[:train_mass.shape[0]], mass[train_mass.shape[0]:]

class CreateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, hidden_layers=1, activation="relu", weight_init="lecun"):
        super().__init__()
        self.linear_non_linear_stack = nn.Sequential().to(device)
        
        modules = []
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
                
        modules.append(nn.Linear(hidden_size, num_classes))
        
        self.linear_non_linear_stack = nn.Sequential(*modules).to(device)

    def forward(self, x):
        x = x.to(device)
        logits = self.linear_non_linear_stack(x)
        return logits
    
def evaluate_loop(dataloader, model, loss_fn, p="True"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).long()  # Move input and target tensors to the same device as the model
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    if p == True:
        print(f"Test Error: \n Accuracy: {(correct):>0.4f}, Avg loss: {test_loss:>8f} \n")
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss

print("Calcification, assessment prediction")
train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(train_calc, test_calc, "a", 64)
model = FNN(5, 512, 6)
model.load_state_dict(torch.load("FNN_fac.pt"))
# print(model.eval())
evaluate_loop(test_loader, model, nn.CrossEntropyLoss(), p=True)

print("Mass, assessment prediction")
train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(train_mass, test_mass, "a", 64)
model = FNN(5, 512, 6)
# for key in torch.load("FNN_fam.pt").keys():
#     print(f"{key} : ", torch.load("FNN_fam.pt")[key].shape)
model.load_state_dict(torch.load("FNN_fam.pt"))
# print(model.eval())
evaluate_loop(test_loader, model, nn.CrossEntropyLoss(), p=True)

print("Calcification, pathology prediction")
train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(train_calc, test_calc, "p", 64)
model = FNN(6, 512, 3, hidden_layers=2)
# for key in torch.load("FNN_fpc.pt").keys():
#     print(f"{key} : ", torch.load("FNN_fpc.pt")[key].shape)
model.load_state_dict(torch.load("FNN_fpc.pt"))
# print(model.eval())
evaluate_loop(test_loader, model, nn.CrossEntropyLoss(), p=True)

print("Mass, pathology prediction")
train_dataset, test_dataset, train_loader, test_loader = get_dataloaders(train_mass, test_mass, "p", 64)
model = FNN(6, 512, 3)
# for key in torch.load("FNN_fpm.pt").keys():
#     print(f"{key} : ", torch.load("FNN_fpm.pt")[key].shape)
model.load_state_dict(torch.load("FNN_fpm.pt"))
# print(model.eval())
evaluate_loop(test_loader, model, nn.CrossEntropyLoss(), p=True)