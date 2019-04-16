# importing necessary libraries

import numpy as np
import pandas as pd
import os
import datetime
import pytz
import random
import time
import copy
import collections
import matplotlib.pyplot as plt

# PyTorch stuff
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch import optim

# Sklearn stuff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
import lightgbm as lgb
import xgboost as xgb
import statsmodels.api as sm

from IPython.display import display_html
from tqdm import tqdm_notebook
from catboost import CatBoostClassifier
from itertools import combinations

import ujson as json

pd.options.mode.chained_assignment = None  # default='warn'

SEED = 42


class PickleHelper:
    # TODO make generic data save/load/fetch method

    PICKLE_PATH = './pickle/'

    def save(self, df, filename):
        if not os.path.exists(self.PICKLE_PATH):
            os.makedirs(self.PICKLE_PATH)

        df.to_pickle(self.PICKLE_PATH + filename + '.pkl')

    def load(self, filename):
        if os.path.exists(self.PICKLE_PATH + filename + '.pkl'):
            return pd.read_pickle(self.PICKLE_PATH + filename + '.pkl')
        return pd.DataFrame()


class ColumnDataProcessor:
    to_scale = False

    def replaceNaNValues(self, A):
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        return A

    def add_team_features(self, df, feature, r_columns, d_columns):
        # TODO use separate df for new features, bundled add and drop
        drop_features = []
        # df['r_total_' + c] = df[r_columns].sum(1)
        # df['d_total_' + c] = df[d_columns].sum(1)
        # df['total_' + c + '_ratio'] = df['r_total_' + c] / df['d_total_' + c]
        # df['total_' + c + '_ratio'] = self.replaceNaNValues(df['total_' + c + '_ratio'])
        # drop_features = drop_features + ['r_total_' + c, 'd_total_' + c]

        df['r_std_' + feature] = df[r_columns].std(1)
        df['d_std_' + feature] = df[d_columns].std(1)
        df['std_' + feature + '_ratio'] = df['r_std_' + feature] / df['d_std_' + feature]
        df['std_' + feature + '_ratio'] = self.replaceNaNValues(df['std_' + feature + '_ratio'])
        drop_features = drop_features + ['r_std_' + feature, 'd_std_' + feature]

        df['r_mean_' + feature] = df[r_columns].mean(1)
        df['d_mean_' + feature] = df[d_columns].mean(1)/df['game_time']
        # df['max_' + feature + '_diff'] = df[r_columns].max(1) - df[d_columns].min(1)
        df['mean_' + feature + '_diff'] = df['r_mean_' + feature] - df['d_mean_' + feature]
        df['mean_' + feature + '_ratio'] = df['r_mean_' + feature] / df['d_mean_' + feature]
        df['mean_' + feature + '_ratio'] = self.replaceNaNValues(df['mean_' + feature + '_ratio'])

        df['r_mean_' + feature + '_per_min'] = df['r_mean_' + feature] / df['game_time']
        df['d_mean_' + feature + '_per_min'] = df['d_mean_' + feature] / df['game_time']
        df['mean_' + feature + '_diff_per_min'] = \
            df['r_mean_' + feature + '_per_min'] - df['d_mean_' + feature + '_per_min']
        # drop_features = drop_features + ['r_mean_' + feature, 'd_mean_' + feature]

        df = df.drop(r_columns, axis=1)
        df = df.drop(d_columns, axis=1)
        df = df.drop(drop_features, axis=1)
        return df

    def fantasy_points_df(self, df, each_player=True):
        for team in 'r', 'd':
            players = f'{team}_mean'
            if (each_player):
                players = [f'{team}{i}' for i in range(1, 6)]  # r1, r2...

            for player in players:
                df[f'{player}_fantasy_points'] = self.fantasy_points(
                    df[f'{player}_kills'],
                    df[f'{player}_deaths'],
                    df[f'{player}_denies'],
                    df[f'{player}_deaths'],
                    df[f'{player}_gold'] / df['game_time'],
                    df[f'{player}_towers_killed'],
                    df[f'{player}_roshans_killed'],
                    df[f'{player}_stuns'],
                    df[f'{player}_teamfight_participation'],
                    df[f'{player}_obs_placed'],
                    df[f'{player}_camps_stacked'],
                    df[f'{player}_rune_pickups'],
                    df[f'{player}_firstblood_claimed']
                )
        return df

    def fantasy_points(self, kills, deaths, last_hits, denies, gold_per_min, towers_killed, roshans_killed, stuns,
                       teamfight_participation, observers_placed, camps_stacked, rune_pickups, firstblood_claimed):
        fantasy_points = (
                    0.3 * kills +
                    (3 - 0.3 * deaths) +
                    0.003 * (last_hits + denies) +
                    0.002 * gold_per_min +
                    towers_killed +
                    roshans_killed +
                    3 * teamfight_participation +
                    0.5 * observers_placed +
                    0.5 * camps_stacked +
                    0.25 * rune_pickups +
                    4 * firstblood_claimed +
                    0.05 * stuns
            )
        return fantasy_points

    # As we see coordinate features (x and y) are quite important. However, I think we need to combine them into one
    # feature.Simplest idea is the distance from the left bottom corner. So, short distances mean near own base,
    # long distances - near the enemy base
    def make_coordinate_features(self, df):
        for team in 'r', 'd':
            players = [f'{team}{i}' for i in range(1, 6)]  # r1, r2...
            for player in players:
                df[f'{player}_distance'] = np.sqrt(df[f'{player}_x'] ** 2 + df[f'{player}_y'] ** 2)
                df.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)
        return df

    # def hot_feat_hero_id(self, df):
    #     for team in ['r', 'd']:
    #         for i in range(1, 6):
    #             df = pd.concat([df, pd.get_dummies(df[f'{team}{i}_hero_id'], prefix=f'{team}{i}_hero_id')], axis=1)
    #
    #     return df

    def hero_id_subset_analyzer(self, text):
        # it takes a string of hero ids (like '1 2 5 4 3') as input
        ids = set()
        for i in range(1, 4):  # we need all subset of lenght 1-3. I think longer combinations are not relevant
            hero_ids = text.split(' ')  # '1 2 5 4 3'-> ['1', '2', '5', '4', '3']
            hero_ids.sort()  # sort them as '1 2 5 4 3' and '3 1 4 5 3' should produce the same set of tokens
            combs = set(
                combinations(hero_ids, i))  # all combinations of length i e.g for 2 are: (1,2), (1,3)... (2,5)... etc
            ids = ids.union(combs)
        ids = {"_".join(item) for item in ids}  # convert from lists to string e.g. (1,2) -> '1_2'
        return ids

    def replace_hero_ids(self, train, test):
        vectorizer = TfidfVectorizer(self.hero_id_subset_analyzer, ngram_range=(1, 1), max_features=1000,
                                     tokenizer=lambda s: s.split())

        full_df = pd.concat([train, test], sort=False)
        train_size = train.shape[0]
        full_df = self.replace_hero_ids_df(full_df, vectorizer)

        # train = self.replace_hero_ids_df(train, vectorizer)
        # test = self.replace_hero_ids_df(test, vectorizer, train=False)

        train = full_df.iloc[:train_size, :]
        test = full_df.iloc[train_size:, :]

        return train, test

    def replace_hero_ids_df(self, df, vectorizer, train=True):

        # ngram range is (1,1) as all combinations are created by analyser
        # 1000 features - I think it's enough to cover all heroes + popular combos

        for team in 'r', 'd':
            players = [f'{team}{i}' for i in range(1, 6)]  # r1, r2,...
            hero_columns = [f'{player}_hero_id' for player in players]  # r1_hero_id,....

            # combine all hero id columns into one
            df_hero_id_as_text = df[hero_columns].apply(lambda row: ' '.join([str(i) for i in row]), axis=1).tolist()

            if train:
                new_cols = pd.DataFrame(vectorizer.fit_transform(df_hero_id_as_text).todense(),
                                        columns=vectorizer.get_feature_names())
            else:
                new_cols = pd.DataFrame(vectorizer.transform(df_hero_id_as_text).todense(),
                                        columns=vectorizer.get_feature_names())

            # add index to vectorized dataset - needed for merge?
            new_cols['match_id_hash'] = df.index.values
            new_cols = new_cols.set_index('match_id_hash').add_prefix(f'{team}_hero_')  # e.g.r_hero_10_21

            # df = pd.merge(df, new_cols)
            # df = pd.merge(df, new_cols, on='match_id_hash')
            df = pd.merge(df, new_cols, left_index=True, right_index=True)
            df.drop(columns=hero_columns, inplace=True)

        return df

    def prepare_data(self, train, target, test, features_list):
        print('prepare_data.. Start')
        train['game_time'] = train['game_time'].apply(lambda x: 1 if x < 60 else x / 60)
        test['game_time'] = test['game_time'].apply(lambda x: 1 if x < 60 else x / 60)

        # r_heroes = [f'r{i}_hero_id' for i in range(1, 6)]
        # d_heroes = [f'd{i}_hero_id' for i in range(1, 6)]

        train = self.make_coordinate_features(train)
        test = self.make_coordinate_features(test)
        # As the distance is also a numeric feature convert it into the team features
        features_list = features_list + ['distance']

        train = self.fantasy_points_df(train)
        test = self.fantasy_points_df(test)
        features_list = features_list + ['fantasy_points']

        print(f'prepare_data.. Adding team features:\n{features_list}')
        for feature in features_list:
            r_columns = [f'r{i}_{feature}' for i in range(1, 6)]
            d_columns = [f'd{i}_{feature}' for i in range(1, 6)]

            train = self.add_team_features(train, feature, r_columns, d_columns)
            test = self.add_team_features(test, feature, r_columns, d_columns)

            if self.to_scale:
                features_to_scale = \
                    ['std_' + feature + '_ratio',
                     'mean_' + feature + '_ratio',
                     'mean_' + feature + '_diff',
                     'r_mean_' + feature,
                     'd_mean_' + feature,
                     'mean_' + feature + '_diff_per_min',
                     'r_mean_' + feature + '_per_min',
                     'd_mean_' + feature + '_per_min',
                     ]
                   # 'total_' + c + '_ratio', 'std_' + c + '_ratio']  # + r_heroes + d_heroes
                scaler = MinMaxScaler()
                train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
                test[features_to_scale] = scaler.transform(test[features_to_scale])


        print('prepare_data.. Replace heroes id')

        train, test = self.replace_hero_ids(train, test)
        # train = self.hot_feat_hero_id(train)
        # test = self.hot_feat_hero_id(test)

        feat_to_drop = ['game_time', 'game_mode', 'lobby_type', 'objectives_len', 'chat_len']  # + r_heroes + d_heroes
        print('prepare_data.. Drop extra columns: {}'.format(feat_to_drop))
        train = train.drop(feat_to_drop, axis=1)
        test = test.drop(feat_to_drop, axis=1)

        return self.prepare_data_simple(train, target, test)

    def prepare_data_simple(self, train, targets, test):
        X = train.reset_index(drop=True)
        y = targets['radiant_win']
        X_test = test.reset_index(drop=True)

        # for col in train.columns:
        #     if train[col].isnull().any():
        #         print(col, train[col].isnull().sum())
        #
        # for col in test.columns:
        #     if test[col].isnull().any():
        #         print(col, test[col].isnull().sum())

        print("\n\nPrepared data frame: ")
        print(X.describe())
        print('Dimensions: train {}, test {}'.format(X.shape, X_test.shape))

        return X, y, X_test


class JsonDataPrepare:
    LABEL = 'json'

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

    FEATURES_LIST = ['kills', 'deaths', 'assists', 'denies', 'gold', 'lh', 'xp', 'health', 'max_health', 'max_mana',
                     'level', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
                     'firstblood_claimed', 'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed',
                     'sen_placed', 'ability_level', 'max_hero_hit', 'purchase_count', 'count_ability_use',
                     'damage_dealt', 'damage_received']

    # In DOTA there are consumble items, which just restore a small amount of hp/mana or teleports you.
    # These items do not affect the outcome of the game, so let's remove it!
    consumable_columns = ['tango', 'tpscroll',
                          'bottle', 'flask',
                          'enchanted_mango', 'clarity',
                          'faerie_fire', 'ward_observer',
                          'ward_sentry']

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
            # adding hero inventory
            row.append((f'{player_name}_items', list(map(lambda x: x['id'][5:], player['hero_inventory']))))

            # row.append((f'{player_name}_items',
            #             list(map(lambda x: x['id'][5:] + f'_hero_{player["hero_id"]}', player['hero_inventory']))))

            # row.append((f'{player_name}_items',
            #             list(map(lambda x: x + f'_hero_{player["hero_id"]}',
            #                      filter(lambda x: x not in self.consumable_columns,
            #                             map(lambda x: x['id'][5:], player['hero_inventory']))))))

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
        pick = PickleHelper()
        df_new_features = pick.load(f'df_train_{self.LABEL}')
        if not df_new_features.empty:
            df_new_targets = pick.load(f'df_targets_{self.LABEL}')
            test_new_features = pick.load(f'df_test_{self.LABEL}')
            print('Dataframes were read from pkl')
        else:
            start = time.time()
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
                features = self.extract_features_csv(match)
                test_new_features.append(features)

            test_new_features = pd.DataFrame.from_records(test_new_features).set_index('match_id_hash')

            pick.save(df_new_features, f'df_train_{self.LABEL}')
            pick.save(df_new_targets, f'df_targets_{self.LABEL}')
            pick.save(test_new_features, f'df_test_{self.LABEL}')
            print(f'Data read in {time.time() - start}')

        print("Original data frame (JSON): ")
        print(df_new_features.shape)
        print(df_new_targets.shape)


        return df_new_features, df_new_targets, test_new_features

    # TODO add pickle save
    # engineering inventory
    def add_inventory_dummies(self, train_df, test_df):
        pick = PickleHelper()
        train = pick.load(f'df_train_inventory_{self.LABEL}')
        if not train.empty:
            train_df = train
            test_df = pick.load(f'df_test_inventory_{self.LABEL}')
            print('Inventory dataframes were read from pkl')
        else:
            start = time.time()
            THRESHOLD_SUM = 50
            full_df = pd.concat([train_df, test_df], sort=False)
            print(f'add_inventory_dummies start.. df: {full_df.shape}, train: {train_df.shape}, test_df: {test_df.shape}')

            train_size = train_df.shape[0]

            for team in 'r', 'd':
                print(f'add_inventory_dummies: {team} team')
                players = [f'{team}{i}' for i in range(1, 6)]

                # teamwise
                item_columns = [f'{player}_items' for player in players]  # r1_items
                d = pd.DataFrame(index=full_df.index)

                for c in item_columns[0:]:
                    dummies = pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0)
                    for column in dummies.columns:
                        if dummies[column].sum() < THRESHOLD_SUM:
                            dummies.drop(column, axis=1, inplace=True)
                    d = d.add(dummies, fill_value=0)

                # print(d.describe())
                # drop temporary inventory list columns
                full_df.drop(columns=item_columns, inplace=True)

                full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)
                # print('Adding items for players of team {}, result DF: {} {}'.format(team, full_df.shape, full_df.shape[1]))

            print('add_inventory_dummies: added {} features'.format(full_df.shape[1] - train_df.shape[1]))

            train_df = full_df.iloc[:train_size, :]
            test_df = full_df.iloc[train_size:, :]

            pick.save(train_df, f'df_train_inventory_{self.LABEL}')
            pick.save(test_df, f'df_test_inventory_{self.LABEL}')

            print(f'Inventory data processed in {time.time() - start}')

        return train_df, test_df

    def prepare_data(self, train_df, y_df, test_df):
        start = time.time()
        pick = PickleHelper()
        train = pick.load(f'X_train_{self.LABEL}')
        if not train.empty:
            target = pick.load(f'y_targets_{self.LABEL}')
            test = pick.load(f'X_test_{self.LABEL}')
            print(f'Prepared data was read from pkl in {time.time() - start}, train: {train.shape}, target: {target.shape}')
        else:
            train_df, test_df = self.add_inventory_dummies(train_df, test_df)
            engineering = ColumnDataProcessor()
            train, target, test = engineering.prepare_data(train_df, y_df, test_df, self.FEATURES_LIST)

            pick.save(train, f'X_train_{self.LABEL}')
            pick.save(target, f'y_targets_{self.LABEL}')
            pick.save(test, f'X_test_{self.LABEL}')
            print(f'Prepare data finished in {time.time() - start}, train: {train.shape}, target: {target.shape}')


        return train, target, test


class CSVDataPrepare:
    LABEL = 'csv'

    def read_data_frame(self):
        pick = PickleHelper()
        df_train_features = pick.load(f'df_train_{self.LABEL}')
        if not df_train_features.empty:
            df_train_targets = pick.load(f'df_targets_{self.LABEL}')
            df_test_features = pick.load(f'df_test_{self.LABEL}')
            print('Dataframes were read from pkl')
        else:
            start = time.time()
            print('reading Dataframes from csv ..')

            PATH_TO_DATA = '../input/'

            # Train dataset
            df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')
            df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')
            # Test dataset
            df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')

            pick.save(df_train_features, f'df_train_{self.LABEL}')
            pick.save(df_train_targets, f'df_targets_{self.LABEL}')
            pick.save(df_test_features, f'df_test_{self.LABEL}')
            print(f'Data read in {time.time() - start}')

        # Check if there is missing data
        print("Original data frame (CSV): ")
        print(df_train_features.shape)
        # print(df_train_features.head())

        return df_train_features, df_train_targets, df_test_features

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
                     'level', 'stuns', 'creeps_stacked', 'camps_stacked', 'rune_pickups',
                     'firstblood_claimed', 'teamfight_participation', 'towers_killed', 'roshans_killed', 'obs_placed',
                     'sen_placed']

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
def train_model_generic(X, X_test, y, params, folds, model_type='lgb', plot_feature_importance=True, averaging='usual',
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

            # display_html(eli5.show_weights(estimator=model,
            #                                feature_names=train_df.columns.values, top=50))

    # TODO pass as a param
    n_fold = 5
    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')
            plt.show()

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


def lgb_model_tunning(X, y, params):
    print('lgb_model_tunning... Start')

    # Create parameters to search
    gridParams = {
        'learning_rate': [0.02],
        'n_estimators': [40],
        'colsample_bytree': [0.66],
        'subsample': [0.7],
        'reg_lambda': [1],
        'reg_alpha': [0.1],
        'num_leaves': [127],
        'min_data_in_leaf': [50],
        #         'lambda_l1': [0],
        #         'lambda_l2': [1],
        'random_state': [SEED],
        'boosting_type': ['gbdt'],
        'objective': ['binary'],
    }
    # Create classifier to use. Note that parameters have to be input manually
    # not as a dict!
    mdl = lgb.LGBMClassifier(
        metric=params['metric'],
        #         metric_freq=params['metric_freq'],
        is_training_metric=params['is_training_metric'],
        max_bin=params['max_bin'],
        #         tree_learner=params['tree_learner'],
        #         bagging_freq=params['bagging_freq'],
        #         min_data_in_leaf=params['min_data_in_leaf'],
        #         min_sum_hessian_in_leaf=params['min_sum_hessian_in_leaf'],
        is_enable_sparse=params['is_enable_sparse'],
        use_two_round_loading=params['use_two_round_loading'],
        is_save_binary_file=params['is_save_binary_file'],
        n_jobs=-1
    )

    # To view the default model params:
    print(mdl.get_params().keys())

    # Create the grid
    grid = GridSearchCV(mdl,
                        gridParams,
                        verbose=0,
                        cv=4,
                        refit='AUC',
                        scoring={'AUC': 'roc_auc'},
                        n_jobs=-1)
    # Run the grid
    grid.fit(X, y)

    # Print the best parameters found
    print('===== Best params =====')
    print(grid.best_params_)
    print(grid.best_score_)

    # Using parameters already set above, replace in the best from the grid search
    params['learning_rate'] = grid.best_params_['learning_rate']
    params['n_estimators'] = grid.best_params_['n_estimators']
    params['colsample_bytree'] = grid.best_params_['colsample_bytree']
    params['subsample'] = grid.best_params_['subsample']
    params['reg_lambda'] = grid.best_params_['reg_lambda']
    params['reg_alpha'] = grid.best_params_['reg_alpha']
    params['num_leaves'] = grid.best_params_['num_leaves']
    params['min_data_in_leaf'] = grid.best_params_['min_data_in_leaf']
    #     params['lambda_l1'] = grid.best_params_['lambda_l1']
    #     params['lambda_l2'] = grid.best_params_['lambda_l2']
    # params['max_bin'] = grid.best_params_['max_bin']
    # params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

    print('Fitting with params: ')
    print(params)
    return params


def lgb_model(X_train, X_test, y_train, tunning=False):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=SEED)

    params = {'boost': 'gbdt',
              'feature_fraction': 0.05,  # handling overfitting
              'max_depth': -1,
              'metric': 'auc',
              'min_data_in_leaf': 50,
              'num_leaves': 10,  # 10, 32, 64
              'num_threads': -1,
              'verbosity': 1,
              'objective': 'binary',
              'learning_rate': 0.005,  # the changes between one auc and a better one gets really small thus a small
              # learning rate performs better

              # 'reg_alpha': 1.2,
              # 'reg_lambda': 1,
              # 'colsample_bytree': 0.66,
              'bagging_freq': 5,  # handling overfitting
              'bagging_fraction': 0.5,  # handling overfitting - adding some noise
              'boost_from_average': 'false',
              'min_sum_hessian_in_leaf': 10.0,
              'tree_learner': 'serial',
              }

    #           'nthread': 3,  # Updated from nthread
    #           'num_leaves': 64,
    #           'learning_rate': 0.05,
    #           'max_bin': 512,
    #           'subsample_for_bin': 200,
    #           'subsample': 1,
    #           'subsample_freq': 1,
    #           'colsample_bytree': 0.8,
    #           'reg_alpha': 5,
    #           'reg_lambda': 10,
    #           'min_split_gain': 0.5,
    #           'min_child_weight': 1,
    #           'min_child_samples': 5,
    #           'scale_pos_weight': 1,
    #           'num_class': 1,

    if tunning:
        params = lgb_model_tunning(X_train, y_train, params)

    oof_lgb, prediction_lgb, scores = train_model_generic(X_train, X_test, y_train,
                                                          params=params,
                                                          folds=folds,
                                                          model_type='lgb',
                                                          plot_feature_importance=True)

    np.save('predictions.pkl', prediction_lgb, allow_pickle=True)

    return oof_lgb, prediction_lgb, scores


def main():
    tunning = False
    start = time.time()
    # data_loader = CSVDataPrepare()
    data_loader = JsonDataPrepare()
    train_df, targets_df, test_df = data_loader.read_data_frame()
    print(f'Data loaded in {time.time() - start}')

    # print(train_df['d5_items'])

    X_train, y_train, X_test = data_loader.prepare_data(train_df, targets_df, test_df)
    print(f'Data prepared in {time.time() - start}')

    oof_lgb, prediction_lgb, scores = lgb_model(X_train, X_test, y_train, tunning)
    print(f'Model predictions in {time.time() - start}')
    write_to_submission_file(prediction_lgb, test_df)


if __name__ == '__main__':
    main()

# CV mean score: 0.8441, std: 0.0051.
#Dimensions: train (39675, 687)


# Dimensions: train (39675, 774), test (10000, 774) '_per_min'