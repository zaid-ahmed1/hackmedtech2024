
from scipy.signal import cheby2,filtfilt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def ppg_preprocessing(signal,sampling_frequency,fL,fH,order):
    signal = (signal-np.min(signal))/(np.max(signal)-np.min(signal))
    ppg_sm_wins = 50
    sm_wins = 10
        ## PPG filtering
    b, a = cheby2(order, 20, [fL,fH], 'bandpass', fs=sampling_frequency)
    ppg_cb2 = filtfilt(b, a, signal)

    win = round(sampling_frequency * ppg_sm_wins/1000)
    B = 1 / win * np.ones(win)
    ppg = filtfilt(B, 1, ppg_cb2)

    ## PPG' filtering
    win = round(sampling_frequency * sm_wins/1000)
    B1 = 1 / win * np.ones(win)
    dx = np.gradient(ppg)
    vpg = filtfilt(B1, 1, dx)

    ## PPG" filtering
    win = round(sampling_frequency * sm_wins/1000)
    B2 = 1 / win * np.ones(win)
    ddx = np.gradient(vpg)
    apg = filtfilt(B2, 1, ddx)

    ## PPG'" filtering
    win = round(sampling_frequency * sm_wins/1000)
    B3 = 1 / win * np.ones(win)
    dddx = np.gradient(apg)
    jpg = filtfilt(B3, 1, dddx)

    return ppg
def ppg_data(ppg_path, fs):
    signal = pd.read_csv(ppg_path, encoding='utf-8')
    signal = np.squeeze(signal.values)
    lower_cutoff_freq=0.5000001 # Lower cutoff frequency (Hz)
    upper_cutoff_freq=12 # Upper cutoff frequency (Hz)
    order=4 # Filter order
    signal = ppg_preprocessing(signal, fs,lower_cutoff_freq,upper_cutoff_freq,order)
    return signal[0:2*fs]

def visualize_ppg_data(signal,fs):
    plt.clf()

    fig, ax = plt.subplots()

    # create time vector
    # create time vector
    t = np.arange(0, len(signal))/fs

    # plot filtered PPG signal
    ax.plot(t, signal)
    ax.set(xlabel = '', ylabel = 'PPG')
    plt.show()
    return fig



def average_sequential(arr, group_size=3):
    # Ensure the array size is a multiple of 3 for reshaping
    trimmed_length = (len(arr) // group_size) * group_size
    reshaped = arr[:trimmed_length].reshape(-1, group_size)
    return reshaped.mean(axis=1)

