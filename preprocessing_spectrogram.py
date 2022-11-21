"""
Created on Thu Mar 10 09:29:30 2022

@author: ahmed
"""
# %% 
import numpy as np
import scipy.io
#from braindecode.datasets import MOABBDataset
sub = "c" 

dataset1 = scipy.io.loadmat('Dataset BCI competition iv\BCIIV_1\BCICIV_calib_ds1{}.mat'.format(sub), struct_as_record=True)
print(dataset1.keys())
# SciPy.io.loadmat does not deal well with Matlab structures, resulting in lots of
# extra dimensions in the arrays. This makes the code a bit more cluttered

sample_rate = dataset1['nfo']['fs'][0][0][0][0]
EEG = dataset1['cnt'].T
nchannels, nsamples = EEG.shape

channel_names = [s[0] for s in dataset1['nfo']['clab'][0][0][0]]
event_onsets = dataset1['mrk'][0][0][0] #['pos']           #[0][0][0]
event_codes = dataset1['mrk'][0][0][1]                   #[0][0][1]
labels = np.zeros((1, nsamples), int)
labels[0, event_onsets] = event_codes

cl_lab = [s[0] for s in dataset1['nfo']['classes'][0][0][0]]
cl1 = cl_lab[0]
cl2 = cl_lab[1]
nclasses = len(cl_lab)
nevents = len(event_onsets)
req_ch = ['C3', 'Cz' ,'C4'] #

# Print some information
print('Shape of EEG:', EEG.shape)
print('Sample rate:', sample_rate)
print('Number of channels:', nchannels)
print('Channel names:', channel_names)
print('Number of events:', len(event_onsets))
print('Event codes:', np.unique(event_codes))
print('Class labels:', cl_lab)
print('Number of classes:', nclasses)
print('Duration of recordings:', (nsamples / sample_rate), 'seconds')

#%%
#===========Plotting the data======================================
# Dictionary to store the trials in, each class gets an entry
trials = {}

# The time window (in samples) to extract for each trial, here 0.5 -- 2.5 seconds
win = np.arange(int(0*sample_rate), int(4*sample_rate))

# Length of the time window
nsamples = len(win)

# Loop over the classes (right, foot)
for cl, code in zip(cl_lab, np.unique(event_codes)):
    
    # Extract the onsets for the class
    cl_onsets = event_onsets[event_codes == code]
    
    # Allocate memory for the trials
    trials[cl] = np.zeros((nchannels, nsamples, len(cl_onsets)))
    
    # Extract each trial
    for i, onset in enumerate(cl_onsets):
        trials[cl][:,:,i] = EEG[:, win+onset]
  
# Some information about the dimensionality of the data (channels x samples x trials)
print('Shape of trials[cl1]:', trials[cl1].shape)
print('Shape of trials[cl2]:', trials[cl2].shape)
#%%
from matplotlib import mlab

def psd(trials):
    '''
    Calculates for each trial the Power Spectral Density (PSD).
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal
    
    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.  
    freqs : list of floats
        Yhe frequencies for which the PSD was computed (useful for plotting later)
    '''
    
    ntrials = trials.shape[2]
    trials_PSD = np.zeros((nchannels, int (1+nsamples/2) , ntrials))

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD
            (PSD, freqs) = mlab.psd(trials[ch,:,trial], NFFT=int(nsamples), Fs=sample_rate)
            trials_PSD[ch, :, trial] = PSD.ravel()
                
    return trials_PSD, freqs

# Apply the function
psd_r, freqs = psd(trials[cl1])
psd_f, freqs = psd(trials[cl2])
trials_PSD = {cl1: psd_r, cl2: psd_f}

import matplotlib.pyplot as plt

def plot_psd(trials_PSD, freqs, chan_ind, chan_lab=None, maxy=None):
    '''
    Plots PSD data calculated with psd().
    
    Parameters
    ----------
    trials : 3d-array
        The PSD data, as returned by psd()
    freqs : list of floats
        The frequencies for which the PSD is defined, as returned by psd() 
    chan_ind : list of integers
        The indices of the channels to plot
    chan_lab : list of strings
        (optional) List of names for each channel
    maxy : float
        (optional) Limit the y-axis to this value
    '''
    plt.figure(figsize=(12,5))
    
    nchans = len(chan_ind)
    
    # Maximum of 3 plots per row
    nrows = int(np.ceil(nchans / 3))
    ncols = min(3, nchans)
    
    # Enumerate over the channels
    for i,ch in enumerate(chan_ind):
        # Figure out which subplot to draw to
        plt.subplot(nrows,ncols,i+1)
    
        # Plot the PSD for each class
        for cl in trials.keys():
            plt.plot(freqs, np.mean(trials_PSD[cl][ch,:,:], axis=-1), label=cl)
    
        # All plot decoration below...
        
        plt.xlim(0,35)
        
        if maxy != None:
            plt.ylim(0,maxy)
    
        plt.grid()
    
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power [micro-volt*2/Hz]')
        if chan_lab == None:
            plt.title('Channel %d' % (ch+1))
        else:
            plt.title(chan_lab[i])

        plt.legend()
        
    plt.tight_layout()
    #plt.savefig('D:\PhD Ain Shams\Dr Seif\GANs\python_ex\BCI_IV_1\spectrogram\PSD .png' , bbox_inches= 'tight', pad_inches= 0) 
    plt.show()
plot_psd(trials_PSD, freqs, [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    chan_lab=['left/ C3', 'center/ Cz', 'right/ C4'], maxy=500)

chan_ind = [channel_names.index(ch) for ch in req_ch]
print ('chan_ind: ',chan_ind)
#%%
#bandpass filter
import scipy.signal 

def bandpass(trials, lo, hi, sample_rate):
    '''
    Designs and applies a bandpass filter to the signal.
    
    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEGsignal
    lo : float
        Lower frequency bound (in Hz)
    hi : float
        Upper frequency bound (in Hz)
    sample_rate : float
        Sample rate of the signal (in Hz)
    
    Returns
    -------
    trials_filt : 3d-array (channels x samples x trials)
        The bandpassed signal
    '''

    # The iirfilter() function takes the filter order: higher numbers mean a sharper frequency cutoff,
    # but the resulting signal might be shifted in time, lower numbers mean a soft frequency cutoff,
    # but the resulting signal less distorted in time. It also takes the lower and upper frequency bounds
    # to pass, divided by the niquist frequency, which is the sample rate divided by 2:
    a, b = scipy.signal.iirfilter(6, [lo/(sample_rate/2.0), hi/(sample_rate/2.0)])

    # Applying the filter to each trial
    ntrials = trials.shape[2]
    trials_filt = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_filt[:,:,i] = scipy.signal.filtfilt(a, b, trials[:,:,i], axis=1)
    
    return trials_filt

# Apply the function
trials_filt = {cl1: bandpass(trials[cl1], 8, 30, sample_rate),
               cl2: bandpass(trials[cl2], 8, 30, sample_rate)}

psd_r, freqs = psd(trials_filt[cl1])
psd_f, freqs = psd(trials_filt[cl2])
trials_PSD = {cl1: psd_r, cl2: psd_f}

plot_psd(trials_PSD, freqs,  [channel_names.index(ch) for ch in ['C3', 'Cz', 'C4']],
    chan_lab=['left', 'center', 'right'],  maxy=500)

#%%
#================== concatenate images =========================
from PIL import Image

def get_concat_v(im1, im2, im3):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    dst.paste(im3, (0, im1.height+im2.height))

    return dst

#%%
from scipy import signal




def stft_data(X, window_size=128, draw=False, cl = 1):
    ''' 
    Short-time-Fourier transform (STFT) function
    INPUTS:  
    X - EEG data (num_trials, num_eeg_electrodes, time_bins) => (channels x samples x trials)
    window_size - fixed # of time bins to analyze frequency components within
    stride - distance between starting position of adjacent windows
    freq - frequency range to obtain spectral power components
    OUTPUTS: 
    X_stft - STFT transformed eeg data (num_trials, num_eeg_electrodes*freq_bins,time_bins,1)
    num_freq - number of frequency bins
    num_time - number of time bins
    '''
    fs = sample_rate
    num_trials = X.shape[2]
    f, t, Zxx = signal.stft(X[0,:,0], fs=fs,  nperseg=window_size, noverlap=1)
    num_freq= f.shape[0]
    num_time= t.shape[0]
    #Z_mean =  np.empty((Zxx.shape[0],Zxx.shape[1]))
    ch_stft= np.empty((int(num_trials), int(num_freq),int(num_time)))
    ch_stft_av= np.empty((int(ch_stft.shape[1]),int(ch_stft.shape[2])))
 
    
    for i in range(num_trials):
        for j in chan_ind:   
            f, t, Zxx = signal.stft(X[j,:,i], fs=fs,  nperseg= window_size, noverlap=1)
            ch_stft[i] = Zxx 
            #print('ch_stft.shape ',ch_stft.shape)
            plt.figure(figsize=(12,5))
            if draw==True:
                for k in range(num_freq):
                    for l in range(num_time):
                        ch_stft_av[k,l] = np.sum(ch_stft[:,k,l]) /100
                 
                #print('ch_stft_av.shape ',ch_stft_av.shape)
                plt.pcolormesh(t, f, np.abs(Zxx),   cmap='jet', shading='gouraud')
                plt.xlim(0,4)
                plt.ylim(8,30)
                plt.axis('off')
                # plt.title('STFT Magnitude_ch:%d' %j )
                # plt.ylabel('Frequency [Hz]')
                # plt.xlabel('Time [sec]')
                plt.savefig('\spectrogram\CL{0}_3ch\cl{0} STFT_{3}_ch {1}_t{2} .png'.format(cl ,j, i, sub), bbox_inches= 'tight', pad_inches= 0)
                plt.show() 
        img26 = Image.open('\spectrogram\CL{0}_3ch\cl{0} STFT_{2}_ch 26_t{1} .png' .format(cl , i, sub)) # Path to image
        img28 = Image.open('\spectrogram\CL{0}_3ch\cl{0} STFT_{2}_ch 28_t{1} .png' .format(cl , i, sub)) # Path to image
        img30 = Image.open('\spectrogram\CL{0}_3ch\cl{0} STFT_{2}_ch 30_t{1} .png' .format(cl , i, sub)) # Path to image
        get_concat_v(img26, img28, img30).save('\spectrogram\sub_{2}\CL{0}\cl{0} ch3_{2}_tr{1}.bmp' .format(cl , i, sub)) 
    

    return ch_stft, f , t
  
#Apply the function
cl1_stft, num_freq, num_time =stft_data(trials_filt[cl1],128,draw=True,cl = 1)
cl2_stft, num_freq, num_time =stft_data(trials_filt[cl2],128,draw=True,cl = 2)
