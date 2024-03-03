import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler

from scores_enum import Score, Engagement, Attraction, Satisfaction

def parse_metrics(raw: list[list]) -> tuple[list, list, list]:
    if len(raw) < 3:
        print(raw)
        raise ValueError

    score = []
    arousal = []
    valence = []
    boring = []

    for inner_list in raw:
        valence.append(inner_list[0][0])
        inner_list[0][1] = 10 - inner_list[0][1]
        boring.append(inner_list[0][1])
        arousal.append(inner_list[0][2])

        score.append(sum(inner_list[0]) / 4)
    
    return (score, arousal, valence, boring)

def get_status(int_list: list[int], label: str) -> str:
    avg_value = round(np.mean(int_list))
    
    if label == "Score":
        return Score(avg_value).name   
    elif label == "Engagement":
        return Engagement(avg_value).name
    elif label == "Attraction":
        return Attraction(avg_value).name
    elif label == "Satisfaction":
        return Satisfaction(avg_value).name
    else:
        raise ValueError("Invalid label")
    
# the most basic training data that includes features from full windows after processing
# with batch size input
def generate_summary_stats_training_data(df):
    # df : unprocessed dataframe (No batch processing)
    # batch_size : indicate processing size (# of seconds)
    # training columns without creating covariance matrices
    train_cols = ['mean_tp9', 'mean_tp10', 'mean_af7', 'mean_af8',
                     'std_tp9', 'std_tp10', 'std_af7', 'std_af8',
                     'vari_tp9', 'vari_tp10', 'vari_af7', 'vari_af8',
                     'max_tp9', 'max_tp10', 'max_af7', 'max_af8',
                     'min_tp9', 'min_tp10', 'min_af7', 'min_af8',
                     'skew_tp9', 'skew_tp10', 'skew_af7', 'skew_af8',
                     'kurt_tp9', 'kurt_tp10', 'kurt_af7', 'kurt_af8']
    train_cols_dict = dict(zip(train_cols, [[] for i in range(len(train_cols))]))    
    # add mean summary stats on all 4 channels
    train_cols_dict['mean_tp9'].append(df['TP9'].mean())
    train_cols_dict['mean_tp10'].append(df['TP10'].mean())
    train_cols_dict['mean_af7'].append(df['AF7'].mean())
    train_cols_dict['mean_af8'].append(df['AF8'].mean())
    
    # add std summary stats on all 4 channels
    train_cols_dict['std_tp9'].append(np.std(df['TP9'], ddof=1))
    train_cols_dict['std_tp10'].append(np.std(df['TP10'], ddof=1))
    train_cols_dict['std_af7'].append(np.std(df['AF7'], ddof=1))
    train_cols_dict['std_af8'].append(np.std(df['AF8'], ddof=1))
    
    # add variance
    train_cols_dict['vari_tp9'].append(np.var(df['TP9'], ddof=1))
    train_cols_dict['vari_tp10'].append(np.var(df['TP10'], ddof=1))
    train_cols_dict['vari_af7'].append(np.var(df['AF7'], ddof=1))
    train_cols_dict['vari_af8'].append(np.var(df['AF8'], ddof=1))
    
    # add min
    train_cols_dict['min_tp9'].append(df['TP9'].min())
    train_cols_dict['min_tp10'].append(df['TP10'].min())
    train_cols_dict['min_af7'].append(df['AF7'].min())
    train_cols_dict['min_af8'].append(df['AF8'].min())
    
    # add max
    train_cols_dict['max_tp9'].append(df['TP9'].max())
    train_cols_dict['max_tp10'].append(df['TP10'].max())
    train_cols_dict['max_af7'].append(df['AF7'].max())
    train_cols_dict['max_af8'].append(df['AF8'].max())
    
    # add skew
    train_cols_dict['skew_tp9'].append(skew(np.array(df['TP9'])))
    train_cols_dict['skew_tp10'].append(skew(np.array(df['TP10'])))
    train_cols_dict['skew_af7'].append(skew(np.array(df['AF7'])))
    train_cols_dict['skew_af8'].append(skew(np.array(df['AF8'])))
    
    # add kurtosis
    train_cols_dict['kurt_tp9'].append(kurtosis(np.array(df['TP9'])))
    train_cols_dict['kurt_tp10'].append(kurtosis(np.array(df['TP10'])))
    train_cols_dict['kurt_af7'].append(kurtosis(np.array(df['AF7'])))
    train_cols_dict['kurt_af8'].append(kurtosis(np.array(df['AF8'])))
   
    output_training_df = pd.DataFrame(train_cols_dict)

    return output_training_df

"""
Muse LSL Example Auxiliary Tools, modified by Painting with Brainwaves repo

These functions perform the lower-level operations involved in buffering,
epoching, and transforming EEG data into frequency bands

@author: Cassani
Muse LSL Tools adapted from https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/utils.py
"""

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')


def epoch(data, samples_epoch, samples_overlap=0):
    """Extract epochs from a time series.

    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]

    Args:
        data (numpy.ndarray or list of lists): data [n_samples, n_channels]
        samples_epoch (int): window length in samples
        samples_overlap (int): Overlap between windows in samples

    Returns:
        (numpy.ndarray): epoched data of shape
    """

    if isinstance(data, list):
        data = np.array(data)

    n_samples, n_channels = data.shape

    samples_shift = samples_epoch - samples_overlap

    n_epochs = int(
        np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs


def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.

    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata

    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            # Initialize feature_matrix
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))

        feature_matrix[i_epoch, :] = compute_band_powers(
            epochs[:, :, i_epoch], fs).T

    return feature_matrix


def get_feature_names(ch_names):
    """Generate the name of the features.

    Args:
        ch_names (list): electrode names

    Returns:
        (list): feature names
    """
    bands = ['delta', 'theta', 'alpha', 'beta']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer