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
import collections

# PyTorch stuff
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim

# Sklearn stuff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm

from tqdm import tqdm_notebook
from catboost import CatBoostClassifier

import ujson as json

SEED = 42


class ColumnDataProcessor:

    def prepare_data_simple(self, train, targets, test):
        X = train.reset_index(drop=True)
        y = targets['radiant_win']
        X_test = test.reset_index(drop=True)
        return X, y, X_test

    def add_feature_average(self, df, c, r_columns, d_columns):
        df['r_total_' + c] = df[r_columns].sum(1)
        df['d_total_' + c] = df[d_columns].sum(1)
        # df['total_' + c + '_ratio'] = df['d_total_' + c].apply(lambda x: 0 if x == 0 else df['r_total_' + c] / x)

        df['r_std_' + c] = df[r_columns].std(1)
        df['d_std_' + c] = df[d_columns].std(1)
        # df['std_' + c + '_ratio'] = df['r_std_' + c] / df['d_std_' + c] if (df['d_std_' + c] != 0) else 0
        #
        df['r_mean_' + c] = df[r_columns].mean(1)
        df['d_mean_' + c] = df[d_columns].mean(1)
        # df['mean_' + c + '_ratio'] = df['r_mean_' + c] / df['d_mean_' + c] if (df['d_mean_' + c] != 0) else 0

        df = df.drop(r_columns, axis=1).reset_index(drop=True)
        df = df.drop(d_columns, axis=1).reset_index(drop=True)
        return df

    def prepare_data(self, train, target, test, features_list):
        for c in features_list:
            r_columns = [f'r{i}_{c}' for i in range(1, 6)]
            d_columns = [f'd{i}_{c}' for i in range(1, 6)]

            train = self.add_feature_average(train, c, r_columns, d_columns)
            test = self.add_feature_average(test, c, r_columns, d_columns)

        return self.prepare_data_simple(train, target, test)


class CSVDataPrepare:

    def read_data_frame(self):
        PATH_TO_DATA = '../input/'

        # Train dataset
        df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')
        df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')
        # Test dataset
        df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')
        # Check if there is missing data
        print("Original data frame: ")
        # print('df_train_features.isnull() {}'.format(df_train_features.isnull().values.any()))
        # print('df_test_features.isnull() {}'.format(df_test_features.isnull().values.any()))
        print(df_train_features.shape)
        return df_train_features, df_train_targets, df_test_features

    def prepareDataOld(self, train, target, test):
        # Let's combine train and test datasets in one dataset.
        # This allows for addding new features for both datasets at the same time.
        df_full_features = pd.concat([train, test])

        # Index to split the training and test data sets
        idx_split = train.shape[0]

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
        player_features = set(f[3:] for f in train.columns[5:])
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
        y_train = target['radiant_win'].map({True: 1, False: 0})

        print(X_train.head())
        print(X_train.describe())

        # splitting whole dataset on train and test
        # X_train = data.loc[:test_index].drop(["y"], axis=1)
        # y_train = data.loc[:test_index]["y"]
        # X_test = data.loc[test_index:].drop(["y"], axis=1)
        # y_test = data.loc[test_index:]["y"]

        return X_train, X_test, y_train

    def prepareValidationTensors(self, X_train, X_test, y_train, test_size=0.2):
        # Perform a train/validation split
        X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train,
                                                                        test_size=test_size,
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
        return dataloaders, X_train_tensor, X_valid_tensor, y_train_tensor, y_valid_tensor, X_test_tensor

    # My idea behind this FE is the following: Let's take gold, for example. Gold earned by each player can't give
    # us a lot of information. But what is we take total gold by the team? Maybe teams with more gold earned usually
    # win. What if we take mean and std of players' gold in a team? Maybe teams where players tend to have similar
    # parameters are more likely to win. Let's try creating these features.
    FEATURES_LIST = ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana',
                     'level', 'x', 'y', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
                     'firstblood_claimed', 'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed',
                     'sen_placed']

    def prepare_data(self, train, target, test):
        engineering = ColumnDataProcessor()
        train, target, test = engineering.prepare_data(train, target, test, self.FEATURES_LIST)

        r_heroes = [f'r{i}_hero_id' for i in range(1, 6)]
        d_heroes = [f'd{i}_hero_id' for i in range(1, 6)]
        feat_to_drop = ['game_time', 'game_mode', 'lobby_type', 'objectives_len', 'chat_len']

        train = train.drop(r_heroes, axis=1).reset_index(drop=True)
        train = train.drop(d_heroes, axis=1).reset_index(drop=True)
        train = train.drop(feat_to_drop, axis=1).reset_index(drop=True)

        test = test.drop(r_heroes, axis=1).reset_index(drop=True)
        test = test.drop(d_heroes, axis=1).reset_index(drop=True)
        test = test.drop(feat_to_drop, axis=1).reset_index(drop=True)

        return train, target, test


class JsonDataPrepare:
    MATCH_FEATURES = [
        ('game_time', lambda m: m['game_time']),
        ('game_mode', lambda m: m['game_mode']),
        ('lobby_type', lambda m: m['lobby_type']),
        ('objectives_len', lambda m: len(m['objectives'])),
        ('chat_len', lambda m: len(m['chat'])),
    ]

    PLAYER_FIELDS = [
        'hero_id',

        'kills',
        'deaths',
        'assists',
        'denies',

        'gold',
        'lh',
        'xp',
        'health',
        'max_health',
        'max_mana',
        'level',

        'x',
        'y',

        'stuns',
        'creeps_stacked',
        'camps_stacked',
        'rune_pickups',
        'firstblood_claimed',
        'teamfight_participation',
        'towers_killed',
        'roshans_killed',
        'obs_placed',
        'sen_placed',
    ]

    def extract_features_csv(self, match):
        row = [
            ('match_id_hash', match['match_id_hash']),
        ]

        for field, f in self.MATCH_FEATURES:
            row.append((field, f(match)))

        for slot, player in enumerate(match['players']):
            if slot < 5:
                player_name = 'r%d' % (slot + 1)
            else:
                player_name = 'd%d' % (slot - 4)

            for field in self.PLAYER_FIELDS:
                column_name = '%s_%s' % (player_name, field)
                row.append((column_name, player[field]))
            row.append((f'{player_name}_ability_level', len(player['ability_upgrades'])))
            row.append((f'{player_name}_max_hero_hit', player['max_hero_hit']['value']))
            row.append((f'{player_name}_purchase_count', len(player['purchase_log'])))
            row.append((f'{player_name}_count_ability_use', sum(player['ability_uses'].values())))
            row.append((f'{player_name}_damage_dealt', sum(player['damage'].values())))
            row.append((f'{player_name}_damage_received', sum(player['damage_taken'].values())))

        return collections.OrderedDict(row)

    def extract_targets_csv(self, match, targets):
        return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
            (field, targets[field])
            for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
        ])

    def read_matches(self, matches_file):
        MATCHES_COUNT = {
            'test_matches.jsonl': 10000,
            'train_matches.jsonl': 39675,
        }
        _, filename = os.path.split(matches_file)
        total_matches = MATCHES_COUNT.get(filename)

        with open(matches_file) as fin:
            for line in tqdm_notebook(fin, total=total_matches):
                yield json.loads(line)

    def read_data_frame(self):
        PATH_TO_DATA = '../input/'
        df_new_features = []
        df_new_targets = []

        for match in self.read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):
            # match_id_hash = match['match_id_hash']
            features = self.extract_features_csv(match)
            targets = self.extract_targets_csv(match, match['targets'])

            df_new_features.append(features)
            df_new_targets.append(targets)

        df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
        df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')

        test_new_features = []
        for match in self.read_matches(os.path.join(PATH_TO_DATA, 'test_matches.jsonl')):
            # match_id_hash = match['match_id_hash']
            features = self.extract_features_csv(match)

            test_new_features.append(features)

        test_new_features = pd.DataFrame.from_records(test_new_features).set_index('match_id_hash')

        print(df_new_features.shape)

        return df_new_features, df_new_targets, test_new_features

    FEATURES_LIST = ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana',
                     'level', 'x', 'y', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
                     'firstblood_claimed', 'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed',
                     'sen_placed', 'ability_level', 'max_hero_hit', 'purchase_count', 'count_ability_use',
                     'damage_dealt', 'damage_received']

    def prepare_data(self, train, target, test):
        engineering = ColumnDataProcessor()
        train, target, test = engineering.prepare_data(train, target, test, self.FEATURES_LIST)

        return train, target, test


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


# model_type = (lgb|xgb|sklearn|glm|cat)
def train_model_generic(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=False, averaging='usual',
                        model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.loc[train_index], X.loc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if model_type == 'lgb':
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)

            model = lgb.train(params,
                              train_data,
                              num_boost_round=20000,
                              valid_sets=[train_data, valid_data],
                              verbose_eval=1000,
                              early_stopping_rounds=200)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_train.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200,
                              verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_train.columns),
                                         ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_train.columns),
                                   ntree_limit=model.best_ntree_limit)

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict_proba(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')

            y_pred = model.predict_proba(X_test)[:, 1]

        if model_type == 'glm':
            model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
            model_results = model.fit()
            model_results.predict(X_test)
            y_pred_valid = model_results.predict(X_valid).reshape(-1, )
            score = roc_auc_score(y_valid, y_pred_valid)

            y_pred = model_results.predict(X_test)

        if model_type == 'cat':
            model = CatBoostClassifier(iterations=20000, learning_rate=0.05, loss_function='Logloss', eval_metric='AUC',
                                       **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                      verbose=False)

            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test)[:, 1]

        oof[valid_index] = y_pred_valid.reshape(-1, )
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values

        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importance()
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    n_fold = len(folds)
    print('!!!!!!!!! n_fold = {}'.format(n_fold))
    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');

            return oof, prediction, feature_importance
        return oof, prediction, scores

    else:
        return oof, prediction, scores


def MLP_model(dataLoader, X_train, y_train, X_test):
    dataloaders, X_train_tensor, X_valid_tensor, y_train_tensor, y_valid_tensor, X_test_tensor \
        = dataLoader.prepareValidationTensors(X_train, X_test, y_train)


def catboost_model(X_train, y_train):
    # Perform a train/validation split
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

    # score, model = train_predict_Catboost(X_train_part, X_valid, y_train_part, y_valid)
    # print('ROC AUC score = {}'.format(score))


def lgb_model(X, X_test, y):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

    params = {'boost': 'gbdt',
              'feature_fraction': 0.05,
              'learning_rate': 0.01,
              'max_depth': -1,
              'metric': 'auc',
              'min_data_in_leaf': 50,
              'num_leaves': 32,
              'num_threads': -1,
              'verbosity': 1,
              'objective': 'binary'
              }
    oof_lgb, prediction_lgb, scores = train_model_generic(X, X_test, y,
                                                          params=params,
                                                          folds=folds,
                                                          model_type='lgb',
                                                          plot_feature_importance=True)

    return oof_lgb, prediction_lgb, scores


def main():
    # data_loader = CSVDataPrepare()
    data_loader = JsonDataPrepare()
    train, targets, test = data_loader.read_data_frame()
    X_train, X_test, y_train = data_loader.prepare_data(train, targets, test)



    print("\n\nPrepared data frame: ")
    print(X_train.columns)
    print(X_train.describe())

    # lgb_model(X_train, X_test, y_train)

    # output_test_data(model, X_train, X_test_tensor)


if __name__ == '__main__':
    main()
