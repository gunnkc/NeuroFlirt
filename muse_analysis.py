import numpy as np                             # Module that simplifies computations on matrices
import pandas as pd                            # Module that helps computations on tabular data
import matplotlib.pyplot as plt                # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data

from time import sleep

from utils import (update_buffer,              # Our own utility functions
                   get_last_data,
                   compute_band_powers)

MUSE_NAME = 'Muse-73C6'

BUFFER_LENGTH = 5
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
INDEX_CHANNEL = [0]

streams = resolve_byprop('type', 'EEG', timeout=2)

if len(streams) == 0:
    raise RuntimeError('Can\'t find EEG stream.')
else:
    print('Found streams: ')
    print(streams)

print("Start acquiring data")
inlet = StreamInlet(streams[0], max_chunklen=12)
eeg_time_correction = inlet.time_correction()

# Get the stream info
info = inlet.info()
fs = int(info.nominal_srate())

# Initialize raw EEG data buffer
eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
filter_state = None  # for use with the notch filter

# Compute the number of epochs in "buffer_length"
n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                            SHIFT_LENGTH + 1))

# Initialize the band power buffer (for plotting)
# bands will be ordered: [delta, theta, alpha, beta]
band_buffer = np.zeros((n_win_test, 4))

print('Press Ctrl-C in the console to break the while loop.')
while (True):
    # Obtain EEG data from the LSL stream
    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    # Only keep the channel we're interested in
    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

    # Update EEG buffer with the new data
    eeg_buffer, filter_state = update_buffer(
        eeg_buffer, ch_data, notch=True,
        filter_state=filter_state)

    # Get newest samples from the buffer
    data_epoch = get_last_data(eeg_buffer,
                                        EPOCH_LENGTH * fs)

    # Compute band powers
    band_powers = compute_band_powers(data_epoch, fs)
    band_buffer, _ = update_buffer(band_buffer,
                                   np.asarray([band_powers]))
    
    data = {
        'alpha': [band_powers[2]],
        'beta': [band_powers[3]],
        'theta': [band_powers[1]],
        'delta': [band_powers[0]]
    }
            
    df = pd.DataFrame(data)
    print(df)
    sleep(5)
