import preprocess as pr
import model as md
import torch
from torch import nn
from torch.optim import Adam, SGD

# Define Processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch Datasets
data_path = './data/data_akbilgic.csv'
data = pr.load_data(data_path)
data_train, data_test = pr.train_test_split(data, 0.3)

# Model Parameters
# An RNN can take as inputs multiple input sequences (e.g, a matrix of n_seq x n_steps)
x_columns = ['ISE.1', 'SP', 'DAX', 'FTSE', 'NIKKEI', 'BOVESPA', 'EU']
y_column = 'ISE.1'
input_size = len(x_columns)
output_size = 1  # y.shape[1]
hidden_dim = 12  # choose arbitrary.
n_layers = 1
batch_size = 20
n_steps = 10

# Fetch Model
model = md.RNN(input_size, output_size, hidden_dim, n_layers)
model.to(device)

# Parameters
criterion = nn.MSELoss()
lr = 0.001
optimizer = Adam(model.parameters(), lr=lr)
epochs = 500
printing_gap = 10
model_path = 'data/best_model.pt'
saved_model_device = torch.device("cpu")

# Train Model
pr.train_loop(model, epochs, optimizer, criterion, batch_size, n_steps, data_train,
              data_test, x_columns, y_column, printing_gap, saved_model_device, model_path, device)
