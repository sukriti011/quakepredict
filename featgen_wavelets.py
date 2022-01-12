
import numpy as np
import pandas as pd
import os
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15

import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import gc
import warnings
warnings.filterwarnings("ignore")

from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve
from scipy import stats
from sklearn.kernel_ridge import KernelRidge
from itertools import product

from tsfresh.feature_extraction import feature_calculators
from joblib import Parallel, delayed
import pywt

# Create a training file with simple derived features

def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]
    return np.mean(change)

class FeatureGenerator(object):
    def __init__(self, dtype, n_jobs=1, chunk_size=None, skiprows=0):
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.skiprows = skiprows
        self.filename = None
        self.n_jobs = n_jobs
        self.test_files = []
        if self.dtype == 'train':
            self.filename = 'train.csv'
            self.total_data = int(629145481 / self.chunk_size)
        else:
            submission = pd.read_csv('sample_submission.csv')
            for seg_id in submission.seg_id.values:
                self.test_files.append((seg_id, 'test/' + seg_id + '.csv'))
            self.total_data = int(len(submission))

    def read_chunks(self):
        if self.dtype == 'train':
            iter_df = pd.read_csv(self.filename, iterator=True, chunksize=self.chunk_size,
                                  # skiprows=range(1, 1 + self.skiprows),
                                  dtype={'acoustic_data': np.float64, 'time_to_failure': np.float64})
            for counter, df in enumerate(iter_df):
                x = df.acoustic_data.values
                y = df.time_to_failure.values[-1]
                seg_id = 'train_' + str(counter)
                del df
                if len(x) == 150000:
                    yield seg_id, x, y
        else:
            for seg_id, f in self.test_files:
                df = pd.read_csv(f, dtype={'acoustic_data': np.float64})
                x = df.acoustic_data.values[-self.chunk_size:]
                del df
                yield seg_id, x, -999
    
    def get_features(self, x, y, seg_id):
        """
        Gets three groups of features: from original data and from reald and imaginary parts of FFT.
        """
        
        x = pd.Series(x)
        main_dict = self.features(x, y, seg_id)
        return main_dict

    def seg_features(self, prefix, x):
        
        peaks = [5, 10, 50]

        feature_dict = dict()
        # basic stats
        # feature_dict[prefix+'mean'] = x.mean()
        feature_dict[prefix+'std'] = x.std()
        feature_dict[prefix+'max'] = x.max()
        #feature_dict[prefix+'min'] = x.min()
            
        # basic stats on absolute values
        feature_dict[prefix+'abs_max'] = np.abs(x).max()
        feature_dict[prefix+'abs_mean'] = np.abs(x).mean()
        feature_dict[prefix+'abs_std'] = np.abs(x).std()
            
        # percentiles on original and absolute values
        # for p in percentiles:
        #    feature_dict[prefix+f'percentile_{p}'] = np.percentile(x, p)
        #    feature_dict[prefix+f'abs_percentile_{p}'] = np.percentile(np.abs(x), p)
        #feature_dict[prefix+'num_crossing_0'] = feature_calculators.number_crossing_m(x, 0)

        for peak in peaks:
            try:
                feature_dict[prefix+f'num_peaks_{peak}'] = feature_calculators.number_peaks(x, peak)
            except:
                feature_dict[prefix+f'num_peaks_{peak}'] = 0

        return feature_dict        
    
    def features(self, x_full, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # create features here
        for i in range(0, 15):
            x = x_full[(1 + i*10000): (i+1)*10000].copy()

            prefix = 'seg{}_'.format(i)
            seg_feature_dict = self.seg_features(prefix, x)

            feature_dict.update(seg_feature_dict)

        wavelet = 'db2'
        level = 4
        order = "freq"  # other option is "normal"
        wp = pywt.WaveletPacket(x_full.copy(), wavelet, 'symmetric', maxlevel=level)
        nodes = wp.get_level(level, order=order)
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)

        for idx in [0, 1, 2]:
            for i in range(0, 10):
                x = values[idx, (1 + i*1000):min((i+1)*1000, values.shape[1])]
                prefix = 'w{}_seg{}_'.format(idx, i)
                seg_feature_dict = self.seg_features(prefix, x)                           
                feature_dict.update(seg_feature_dict)
        
        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs)(delayed(self.get_features)(x, y, s)
                                            for s, x, y in tqdm(self.read_chunks(), total=self.total_data))
        for r in res:
            feature_list.append(r)
        return pd.DataFrame(feature_list)

def main(argv):

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skiprows', type=int, default=0)
    parser.add_argument('-o', '--outputPath', type=str, default='output')
    parser.add_argument('--test', action="store_true", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.outputPath):
        os.makedirs(args.outputPath)   
    
    training_fg = FeatureGenerator(dtype='train', n_jobs=20, chunk_size=150000, skiprows=args.skiprows)
    training_data = training_fg.generate()
    training_data.to_csv(os.path.join(args.outputPath, 'train_data.csv'), index=False)
    if args.test:
        test_fg = FeatureGenerator(dtype='test', n_jobs=20, chunk_size=150000)
        test_data = test_fg.generate()
        test_data.to_csv(os.path.join(args.outputPath, 'test_data.csv'), index=False)

    X = training_data.drop(['target', 'seg_id'], axis=1)
    if args.test:
        X_test = test_data.drop(['target', 'seg_id'], axis=1)
        test_segs = test_data.seg_id
    y = training_data.target
    
    # Fixing missing values
    means_dict = {}
    for col in X.columns:
        if X[col].isnull().any():
            print(col)
            mean_value = X.loc[X[col] != -np.inf, col].mean()
            X.loc[X[col] == -np.inf, col] = mean_value
            X[col] = X[col].fillna(mean_value)
            means_dict[col] = mean_value

    if args.test:
        for col in X_test.columns:
            if X_test[col].isnull().any():
                X_test.loc[X_test[col] == -np.inf, col] = means_dict[col]
                X_test[col] = X_test[col].fillna(means_dict[col])

    X.to_csv(os.path.join(args.outputPath, 'train_features.csv'), index=False)
    if args.test:
        X_test.to_csv(os.path.join(args.outputPath, 'test_features.csv'), index=False)
    y.to_csv(os.path.join(args.outputPath, 'y.csv'), index=False)

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main(sys.argv[1:])
