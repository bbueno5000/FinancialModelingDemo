"""
Taken from Frans Slothoubers post on the contest discussion forum:
    https://www.kaggle.com/slothouber/two-sigma-financial-modeling/kagglegym-emulation
"""
import numpy
import os
import pandas
import sklearn.metrics

def r_score(y_true, y_pred, sample_weight=None, multioutput=None):
    """
    DOCSTRING
    """
    r2 = sklearn.metrics.r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    r = (numpy.sign(r2)*numpy.sqrt(numpy.abs(r2)))
    if r <= -1:
        return -1
    else:
        return r

class Environment:
    """
    DOCSTRING
    """
    def __init__(self):
        with pandas.HDFStore("/content/drive/My Drive/train.h5", "r") as hfdata:
            self.timestamp = 0
            fullset = hfdata.get("train")
            self.unique_timestamp = fullset["timestamp"].unique()
            n = len(self.unique_timestamp)
            i = int(n / 2)
            timesplit = self.unique_timestamp[i]
            self.n = n
            self.unique_idx = i
            self.train = fullset[fullset.timestamp < timesplit]
            self.test = fullset[fullset.timestamp >= timesplit]
            self.full = self.test.loc[:, ['timestamp', 'y']]
            self.full['y_hat'] = 0.0
            self.temp_test_y = None

    def __str__(self):
        return "Environment()"

    def reset(self):
        """
        DOCSTRING
        """
        timesplit = self.unique_timestamp[self.unique_idx]
        self.unique_idx = int(self.n / 2)
        self.unique_idx += 1
        subset = self.test[self.test.timestamp == timesplit]
        # reset index to conform to kaggle gym
        target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
        self.temp_test_y = target['y']
        target.loc[:, 'y'] = 0.0  # set the prediction column to zero
        # changed bounds to 0:110 from 1:111 to mimic the behavior of api
        features = subset.iloc[:, :110].reset_index(drop=True)
        observation = Observation(self.train, target, features)
        return observation

    def step(self, target):
        """
        DOCSTRING
        """
        timesplit = self.unique_timestamp[self.unique_idx-1]
        # Since full and target have a different index we need
        # to do a _values trick here to get the assignment working
        y_hat = target.loc[:, ['y']]
        self.full.loc[self.full.timestamp == timesplit, ['y_hat']] = y_hat._values
        if self.unique_idx == self.n:
            done = True
            observation = None
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            score = r_score(self.full['y'], self.full['y_hat'])
            info = {'public_score': -score}
        else:
            reward = r_score(self.temp_test_y, target.loc[:, 'y'])
            done = False
            info = {}
            timesplit = self.unique_timestamp[self.unique_idx]
            self.unique_idx += 1
            subset = self.test[self.test.timestamp == timesplit]
            # reset index to conform to kaggle gym
            target = subset.loc[:, ['id', 'y']].reset_index(drop=True)
            self.temp_test_y = target['y']
            target.loc[:, 'y'] = 0 # set the prediction column to zero
            # column bound change on the subset
            # reset index to conform to kaggle gym
            features = subset.iloc[:, 0:110].reset_index(drop=True)
            observation = Observation(self.train, target, features)
        return observation, reward, done, info

class Observation:
    """
    DOCSTRING
    """
    def __init__(self, train, target, features):
        self.train = train
        self.target = target
        self.features = features
