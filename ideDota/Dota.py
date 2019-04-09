# importing necessary libraries

import numpy as np
import pandas as pd
import os
import datetime
import pytz
import random
import time
import matplotlib.pyplot as plt
import copy
# import seaborn as sns

# PyTorch stuff
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim

# Sklearn stuff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import catboost
from catboost import CatBoostClassifier

SEED = 42


def readDataframe():
    PATH_TO_DATA = '../input/'

    # Train dataset
    df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'),
                                    index_col='match_id_hash')
    df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'),
                                   index_col='match_id_hash')

    # Test dataset
    df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'),
                                   index_col='match_id_hash')

    # Check if there is missing data
    print('df_train_features.isnull() {}'.format(df_train_features.isnull().values.any()))
    print('df_test_features.isnull() {}'.format(df_test_features.isnull().values.any()))

    print(df_train_features.head())

    return df_train_features, df_train_targets, df_test_features


def prepareData(df_train_features, df_train_targets, df_test_features):
    # Let's combine train and test datasets in one dataset.
    # This allows for addding new features for both datasets at the same time.
    df_full_features = pd.concat([df_train_features, df_test_features])

    # Index to split the training and test data sets
    idx_split = df_train_features.shape[0]

    # That is,
    # df_train_features == df_full_features[:idx_split]
    # df_test_features == df_full_features[idx_split:]

    df_full_features.drop(['game_time', 'game_mode', 'lobby_type', 'objectives_len', 'chat_len'],
                          inplace=True, axis=1)

    # Clearly the hero_id is a categorical feature, so let's one-hot encode it. Note that according to wiki there are
    # 117 heroes, however in our dataset there are 116 heroes with ids 1, 2, ..., 114, 119, 120.
    # You will get the same result for all teams and players, here I use r1.
    np.sort(np.unique(df_full_features['r1_hero_id'].values.flatten()))

    for t in ['r', 'd']:
        for i in range(1, 6):
            df_full_features = pd.get_dummies(df_full_features, columns=[f'{t}{i}_hero_id'])
    #         df_full_features = pd.concat([df_full_features,
    #           pd.get_dummies(df_full_features[f'{t}{i}_hero_id'], prefix=f'{t}{i}_hero_id')], axis=1)

    # Finally let's scale the player-features that have relatively large values, such as gold, lh, xp etc.
    player_features = set(f[3:] for f in df_train_features.columns[5:])
    features_to_scale = []
    for t in ['r', 'd']:
        for i in range(1, 6):
            for f in player_features - {'hero_id', 'firstblood_claimed', 'teamfight_participation'}:
                features_to_scale.append(f'{t}{i}_{f}')
    df_full_features_scaled = df_full_features.copy()
    df_full_features_scaled[features_to_scale] = MinMaxScaler().fit_transform(
        df_full_features_scaled[features_to_scale])

    df_full_features_scaled.head()
    df_full_features_scaled.max().sort_values(ascending=False).head(12)

    # Let's construct X and y arrays.
    X_train = df_full_features_scaled[:idx_split]
    X_test = df_full_features_scaled[idx_split:]
    y_train = df_train_targets['radiant_win'].map({True: 1, False: 0})

    print(X_train.head())

    # splitting whole dataset on train and test
    # X_train = data.loc[:test_index].drop(["y"], axis=1)
    # y_train = data.loc[:test_index]["y"]
    # X_test = data.loc[test_index:].drop(["y"], axis=1)
    # y_test = data.loc[test_index:]["y"]

    return X_train, X_test, y_train


def prepareValidationData(X_train, X_test, y_train, test_size=0.2):
    # Perform a train/validation split
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=test_size,
                                                                    random_state=SEED)

    # Convert to pytorch tensors
    X_train_tensor = torch.from_numpy(X_train_part.values).float()
    X_valid_tensor = torch.from_numpy(X_valid.values).float()
    y_train_tensor = torch.from_numpy(y_train_part.values).float()
    y_valid_tensor = torch.from_numpy(y_valid.values).float()
    X_test_tensor = torch.from_numpy(X_test.values).float()

    # Create the train and validation dataloaders
    train_dataset = data.TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = data.TensorDataset(X_valid_tensor, y_valid_tensor)

    dataloaders = {'train': data.DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=2),
                   'valid': data.DataLoader(valid_dataset, batch_size=1000, shuffle=False, num_workers=2)}
    return dataloaders, X_valid_tensor, y_valid, X_test_tensor


class MLP(nn.Module):
    ''' Multi-layer perceptron with ReLu and Softmax.

    Parameters:
    -----------
        n_input (int): number of nodes in the input layer
        n_hidden (int list): list of number of nodes n_hidden[i] in the i-th hidden layer
        n_output (int):  number of nodes in the output layer
        drop_p (float): drop-out probability [0, 1]
        random_state (int): seed for random number generator (use for reproducibility of result)
    '''

    def __init__(self, n_input, n_hidden, n_output, drop_p, random_state=SEED):
        super().__init__()
        self.random_state = random_state
        set_random_seed(SEED)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in zip(n_hidden[:-1], n_hidden[1:])])
        self.output_layer = nn.Linear(n_hidden[-1], n_output)
        self.dropout = nn.Dropout(p=drop_p)  # method to prevent overfitting

    def forward(self, X):
        ''' Forward propagation -- computes output from input X.
        '''
        for h in self.hidden_layers:
            X = F.relu(h(X))
            X = self.dropout(X)
        X = self.output_layer(X)
        return torch.sigmoid(X)

    def predict_proba(self, X_test):
        return self.forward(X_test).detach().squeeze(1).numpy()


def set_random_seed(rand_seed=SEED):
    ''' Helper function for setting random seed. Use for reproducibility of results'''
    if type(rand_seed) == int:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(rand_seed)
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)


def write_to_submission_file(predicted_labels, df_test_features):
    df_submission = pd.DataFrame({'radiant_win_prob': predicted_labels},
                                 index=df_test_features.index)

    submission_filename = 'submission_{}.csv'.format(
        datetime.datetime.now(tz=pytz.timezone('Europe/Athens')).strftime('%Y-%m-%d_%H-%M-%S'))

    df_submission.to_csv(submission_filename)

    print('Submission saved to {}'.format(submission_filename))


def output_test_data(mlp, X_train, X_test_tensor):
    # Save
    torch.save(mlp.state_dict(), 'mlp.pth')

    # Load
    mlp = MLP(n_input=X_train.shape[1], n_hidden=[200, 100], n_output=1, drop_p=0.4)
    mlp.load_state_dict(torch.load('mlp.pth'))
    mlp.eval()
    mlp_pred = mlp.predict_proba(X_test_tensor)

    write_to_submission_file(mlp_pred)


def trainMPL(model, epochs, criterion, optimizer, scheduler, dataloaders, verbose=False):
    '''
    Train the given model...

    Parameters:
    -----------
        model: model (MLP) to train
        epochs (int): number of epochs
        criterion: loss function e.g. BCELoss
        optimizer: optimizer e.g SGD or Adam
        scheduler: learning rate scheduler e.g. StepLR
        dataloaders: train and validation dataloaders
        verbose (boolean): print training details (elapsed time and losses)

    '''
    t0_tot = time.time()

    set_random_seed(model.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}...')
    model.to(device)

    # Best model weights (deepcopy them because model.state_dict() changes during the training)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    losses = {'train': [], 'valid': []}

    for epoch in range(epochs):
        t0 = time.time()
        print(f'============== Epoch {epoch + 1}/{epochs} ==============')
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                if verbose: print(f'lr: {scheduler.get_lr()}')
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            for ii, (X_batch, y_batch) in enumerate(dataloaders[phase], start=1):
                # Move input and label tensors to the GPU
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Reset the gradients because they are accumulated
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(X_batch).squeeze(1)  # forward prop
                    loss = criterion(outputs, y_batch)  # compute loss
                    if phase == 'train':
                        loss.backward()  # backward prop
                        optimizer.step()  # update the parameters

                running_loss += loss.item() * X_batch.shape[0]

            ep_loss = running_loss / len(dataloaders[phase].dataset)  # average loss over an epoch
            losses[phase].append(ep_loss)
            if verbose: print(f' ({phase}) Loss: {ep_loss:.5f}')

            # Best model by lowest validation loss
            if phase == 'valid' and ep_loss < best_loss:
                best_loss = ep_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        if verbose: print(f'\nElapsed time: {round(time.time() - t0, 3)} sec\n')

    print(f'\nTraining completed in {round(time.time() - t0_tot, 3)} sec')

    # Load the best model weights to the trained model
    model.load_state_dict(best_model_wts)
    model.losses = losses
    model.to('cpu')
    model.eval()
    return model


def plot_losses(train_losses, val_losses):
    y = [train_losses, val_losses]
    c = ['C7', 'C9']
    labels = ['Train loss', 'Validation loss']
    # Plot train_losses and val_losses wrt epochs
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = list(range(1, len(train_losses) + 1))
    for i in range(2):
        ax.plot(x, y[i], lw=3, label=labels[i], color=c[i])
        ax.set_xlabel('Epoch', fontsize=16)
        ax.set_ylabel('Loss', fontsize=16)
        ax.set_xticks(range(0, x[-1] + 1, 2))
        ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def train_predict_MLP(dataloaders, X_train, X_valid_tensor, y_valid):
    mlp = MLP(n_input=X_train.shape[1], n_hidden=[200, 100], n_output=1, drop_p=0.4)

    criterion = nn.BCELoss()  # Binary cross entropy
    optimizer = optim.Adam(mlp.parameters(), lr=0.01,
                           weight_decay=0.005)  # alternatevily torch.optim.SGD(mlp.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 12
    trainMPL(mlp, epochs, criterion, optimizer, scheduler, dataloaders, verbose=True)
    plot_losses(mlp.losses['train'], mlp.losses['valid'])
    score = roc_auc_score(y_valid.values, mlp.predict_proba(X_valid_tensor))
    return score, mlp


def train_predict_Catboost(dataloaders, X_train, X_valid_tensor, y_valid):
    model = CatBoostClassifier(iterations=200,
                               task_type="GPU",
                               verbose=1)

    model.fit(dataloaders['train'])

    y_predict = model.predict(X_valid_tensor)

    score = roc_auc_score(y_valid.values, y_predict)
    return score, model


def main():
    df_train_features, df_train_targets, df_test_features = readDataframe();
    X_train, X_test, y_train = prepareData(df_train_features, df_train_targets, df_test_features)
    dataloaders, X_valid_tensor, y_valid, X_test_tensor = prepareValidationData(X_train, X_test, y_train)
    score, model = train_predict_Catboost(dataloaders, X_train, X_valid_tensor, y_valid)
    print('ROC AUC score = {}'.format(score))

    # output_test_data(model, X_train, X_test_tensor)


if __name__ == '__main__':
    main()
