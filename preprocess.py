from torchvision import datasets, transforms
import torch
import copy
import numpy as np
import pandas as pd

# Used for plotting and display of figures
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def load_data(path):
    return pd.read_csv(path)


def train_test_split(df, test=0.3):
    idx = int(df.shape[0] * (1 - test))
    return df.iloc[:idx, :], df.iloc[idx:, :]


def next_stock_batch(batch_size, n_steps, df, x_columns, y_column):
    # Receives the number of samples (batch_size) of size (n_steps) to extract
    # from the time series, and outputs such a sample

    t_min = 0
    t_max = df.shape[0]

    x = np.zeros((batch_size, n_steps, len(x_columns)))
    y = np.zeros((batch_size, n_steps, 1))

    starting_points = np.random.randint(0, t_max - n_steps - 1, size=batch_size)

    # We create the batches for x using all time series between t and t+n_steps    
    # We create the batches for y using only one time series between t+1 and t+n_steps+1
    df_x = df[x_columns].values
    df_y = df[y_column].values
    for i, sp in enumerate(starting_points):
        x[i, :], y[i] = df_x[sp:sp + n_steps, :], df_y[sp + 1:sp + n_steps + 1].reshape(-1, n_steps, 1)

    return x, y


def train_loop(model, epochs, optimizer, criterion, batch_size, n_steps, df_train, df_test, x_columns, y_column,
               printing_gap, saved_model_device, model_path, device):
    train_loss = []
    test_loss = []
    least_loss = np.inf

    for epoch in range(1, epochs + 1):

        model.train()
        seq, target = next_stock_batch(batch_size, n_steps, df_train, x_columns, y_column)
        seq, target = torch.from_numpy(seq).float().to(device), torch.from_numpy(target).float().to(device)

        optimizer.zero_grad()
        output, _ = model.forward(seq)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            seq, target = next_stock_batch(batch_size, n_steps, df_test, x_columns, y_column)
            seq, target = torch.from_numpy(seq).float().to(device), torch.from_numpy(target).float().to(device)

            output, _ = model.forward(seq)
            loss_test = criterion(output, target)

        train_loss.append(loss.item())
        test_loss.append(loss_test.item())

        # Save best model
        if loss < least_loss:
            least_loss = loss

            best_model_state = copy.deepcopy(model)
            best_model_state.to(saved_model_device)
            torch.save(best_model_state, model_path)

        if epoch % printing_gap == 0:
            print('Epoch: {}/{}\t.............'.format(epoch, epochs), end=' ')
            print("Train Loss: {:.4f}".format(loss.item()), end=' ')
            print("Test Loss: {:.4f}".format(loss_test.item()))

    plt.plot(train_loss, label="Train Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.xlabel(" Iteration ")
    plt.ylabel("Loss value")
    plt.legend(loc="upper left")
    plt.show()
