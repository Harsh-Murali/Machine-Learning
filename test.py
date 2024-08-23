import torch
import torch.nn as nn

class MatrixFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MatrixFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten the matrix input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# create an instance of the MatrixFNN model
input_size = 3 * 3  # size of the input matrix
hidden_size = 16
output_size = 9  # we want to predict an integer between 1 and 9
model = MatrixFNN(input_size, hidden_size, output_size)

# example usage
input_matrix = torch.randn(1, 3, 3)  # create a random 3x3 matrix input
output_logits = model(input_matrix)  # get the logits for the 9 output neurons
predicted_integer = torch.argmax(output_logits) + 1  # get the predicted integer value
print(predicted_integer.item())  # print the predicted integer value as a scalar
