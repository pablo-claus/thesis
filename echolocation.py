import numpy as np
from scipy.signal import correlate, chirp, find_peaks, fftconvolve, general_gaussian
#from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import sounddevice
import time
from scipy.io import wavfile
from datetime import datetime
import os
import pickle
from scipy.signal import butter, lfilter

# This is the code for the 'Development of acoustic echolocation in an unmanned aerial vehicle' thesis

# This Python3 file contains several methods to be able to record and estimate the distance based on reflected sweep signals

# The main method is called 'echolocation()'.
# It can be run to execute the echolocation algorithm automatically. To enable all functionalities, a couple of helper classes are defined as well


### Some more details on the drone set-up and positions of the speaker and mics
#
# Channels:
#
# 1,2 are front facing i.e. the same side along with the speaker. Mics 9&10 are on the same side
#
# 1___9
# |   | <-Speaker at the centre
# 10__2  
#
# Mics 3 up to 8 are distributed two on every side, clockwise from the drone
#
# i___
# |   |
# |___(i+1)
#
#
# take into account that in Python the numbering of the channels start at 0
# naming convention drone by facing the drone
#
#     drone sides:
#
#       BACK		
#  	 ___________
# L	|	    	| R
# E	|			| I
# F	|			| G
# T	|  speaker	| H
#	|___________| T
#	    FRONT
#

# Channel mics positions
ch_frontside_topleft = 0
ch_frontside_bottomright = 1
ch_frontside_topright = 8
ch_frontside_bottomleft = 9

ch_rightside_topleft = 2
ch_rightside_bottomright = 3

ch_backside_topleft = 4
ch_backside_bottomright = 5

ch_leftside_topleft = 6
ch_leftside_bottomright = 7


# Extra global variables

sos = 343 # speed of sound at 20°C
nb_channels = 1#10
ch_selection = 0#9
delay_speaker_mic = 0.0625 #cut off the first delay_speaker_mic*fs samples from the recording
offset = 31

# get the date, for easy file management
today = datetime.now()


# Main function, get the distance estimation several times and return the median
def echolocation(measurements_taken = 5, enable_print_msg = True):
    
    fs = 32000
    f_start = 9000 
    f_end = 12000 
    chirp_duration = 0.004
    detection_duration = 0.1 # echo limit
    
    
    estimations = np.array([])
    count = 0
    while count < measurements_taken:
        distance_estimation = play_and_calculate(chirp_duration, f_start, f_end, detection_duration, fs)
        if distance_estimation != -1:
            estimations = np.append(estimations, distance_estimation)
            if enable_print_msg:
                print('Estimation #' + str(count) + ' successfully made')
            count+=1
            
        else:
            if enable_print_msg:
                print('Something went wrong with estimation #' + str(count) + ' another attempt is made')
            
    median_distance = np.median(estimations)
    print ('Distance = ' + str(median_distance) + ' metres')
    return 


# Method that plays sweeps and calculates the distance to a wall based on this data
def play_and_calculate(chirp_duration, f_start, f_end, detection_duration, fs, description=''):
    
    chirp_template, recorded_chirp = record_sweep(fs, f_start, f_end, chirp_duration,detection_duration, description=description, save_wav=False, )
    calc_distance = calculate_distance(template=chirp_template, inputsignal = recorded_chirp[:, ch_selection], duration_template = chirp_duration, samplerate=fs, f0=f_start, f1=f_end, domain = 'time', display_plots = True, date=today, description=description)
              
    return calc_distance


# Method that performs actual distance calculation based on the results of the matched filter's cross correlation
def calculate_distance(template, inputsignal, duration_template, samplerate, domain, date, f0, f1, description='', display_plots = False, datebased_filename=True, bandpass = True):
    date_folder = create_date_folder(date)
    if bandpass:
        filtered = butter_bandpass_filter(data=inputsignal, lowcut=8500, highcut=12500, fs=samplerate, order=5)
        inputsignal = filtered
    correl = corr(template, inputsignal, samplerate, domain)
    peak = get_reflected_peak(correl, len(template))
    if peak == -1:
        return -1
    else:
        calc_distance = get_peak_distance(peak, samplerate)
        if display_plots: 

            if datebased_filename:
                png_name = date_folder +  get_date_based_filename(date)  + '_' + domain +  '.png'
            else:
                png_folder = './png/' + description + '/'
                if not os.path.exists(png_folder):
                    os.makedirs(png_folder)
                png_name = png_folder + format_nb_digits(round(duration_template*1000),2) + 'ms_' + str(round(f0/1000)) + 'k_' + format_nb_digits(round(f1/1000),2) + 'kHz_chirp_' +  description
            
            if description != '':
                description = '[' + description + '] '      
                
            plt.plot(correl)
            plt.plot(peak, correl[peak], "x")
            plt.title('CC with '+ str(round(duration_template*1000)) + 'ms ' + str(round(f0/1000)) + 'k-' + str(round(f1/1000)) + 'kHz chirp, calc. in ' + domain + ' domain at fs= ' + str(round(samplerate/1000,1)) + 'kHz \n' + description + str(round(calc_distance, 2)) + ' metres')
            plt.xlabel('samples')
            plt.ylabel('|Amplitude|')
            plt.savefig(png_name)
            #plt.close()
            #plt.close()
        return calc_distance

# Convert the peak instance to a distance
def get_peak_distance(peak, samplerate):
    bp_sample_offset = 12
    peak -= bp_sample_offset
    peak += offset
    peak_delay = peak/samplerate
    distance = peak_delay*sos
    return round(distance/2,2) # return a single distance from the drone to an object

# Based on the cross correlation, get the peak of the reflected wave
def get_reflected_peak(corr, template_size):
    
    peak_value= np.max(corr)
    peak_instance , _ = find_peaks(corr, height = peak_value)
    return peak_instance[0]
    
 # Calculate the cross correlation between a sweet template and a recorded sweep   
def corr(template, inputsignal, samplerate, domain):
    
    # .shape return a tuple with first element the amount of rows, recond the amount of columns. If the amount of coloms = 1, it will only return amount of rows, so len tuple == 1
    assert len(inputsignal.shape) == 1, "The input signal should have only 1 channel"
    assert len(template.shape) == 1, "The input signal should have only 1 channel"
    assert domain=='time' or domain == 'freq', "Enter if domain should be 'time' or 'freq'"
        
    if domain == 'time':
        corr = correlate(inputsignal, template)
        corr = abs(corr)
    else:
        corr = fftconvolve(template, inputsignal, 'full')
        corr = abs(corr)
        
    return corr

# General method that records a sweep signal
def record_sweep(fs, f_start, f_end, chirp_duration,detection_duration, description='', save_wav=True):
    
    # first incorporate delay between speaker and microphone
    add_sample_delay = 100 + round(chirp_duration*fs) + offset
    detection_duration = detection_duration+delay_speaker_mic+ round(add_sample_delay/fs)
    
    if description != '':
        description = '_'+description.replace(" ", "_")
    t_template = np.linspace(0, chirp_duration, round(chirp_duration*fs))
    chirp_template = chirp(t_template, f0=f_start, f1=f_end, t1=chirp_duration, method='linear')
    
    zeros_padded_front = 0
    zeros_padded_back = round(detection_duration*fs)-zeros_padded_front-round(chirp_duration*fs)
    chirp_zero_padded = np.concatenate([np.zeros(zeros_padded_front), chirp_template, np.zeros(zeros_padded_back)])
    
    global today
    today = datetime.today()
    date_folder = create_date_folder(today)
    recorded_chirp = sounddevice.playrec(chirp_zero_padded, samplerate = fs, channels = nb_channels)#, blocking = True)
	#sounddevice.wait()
	
    time.sleep(3*len(chirp_zero_padded)/fs)

    # cut off delay and length of emitted signal
    
    recorded_chirp = recorded_chirp[(round(delay_speaker_mic*fs) + add_sample_delay):,:]
    
    if save_wav:
        wav_specifications = format_nb_digits(round(chirp_duration*1000),2) + 'ms_' + str(round(f_start/1000)) + 'k_' + str(round(f_end/1000)) + 'kHz_chirp'
        wav_name = date_folder + get_date_based_filename(today) + '_' + wav_specifications + description
        wavfile.write(wav_name + '.wav', fs, recorded_chirp)
    
    return [chirp_template, recorded_chirp]
        
# General method that records a white noise signal
def record_noise(fs, noise_duration,detection_duration, description='', save_wav=True):
    
    # first incorporate delay between speaker and microphone
    detection_duration = detection_duration+delay_speaker_mic
    
    if description != '':
        description = '_'+description.replace(" ", "_")
    noise_template = np.random.normal(0,1,size=round(noise_duration*fs))
    
    
    zeros_padded_front = 0
    zeros_padded_back = round(detection_duration*fs)-zeros_padded_front-round(noise_duration*fs)
    noise_zero_padded = np.concatenate([np.zeros(zeros_padded_front), noise_template, np.zeros(zeros_padded_back)])
    
    # adjust the today datetime variable, the result should be global!!!
    global today
    today = datetime.today()
    date_folder = create_date_folder(today)
    recorded_noise = sounddevice.playrec(noise_zero_padded, samplerate = fs, channels = nb_channels)#, blocking = True)
    time.sleep(2*len(noise_zero_padded)/fs)
    recorded_noise = recorded_noise[round(delay_speaker_mic*fs):,:]
    
    if save_wav:
        wav_specifications = format_nb_digits(round(noise_duration*1000),2) + 'ms_whitenoise'
        wav_name = date_folder + get_date_based_filename(today) + '_' + wav_specifications + description
        wavfile.write(wav_name + '.wav', fs, recorded_noise)
    
    return [noise_template, recorded_noise]

# bandpass code
# https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y





### function for file analysis
#    
  
def analyse_matched(wavname, f0, f1, signal_duration,description):
    fs, wave = wavfile.read(wavname)
    outF = open('Errors_' + description +'.txt', "w")
    channels = wave.shape[1]
    if description != '':
        description = '_'+description.replace(" ", "_")
    t_template = np.linspace(0, signal_duration, signal_duration*fs)
    chirp_template = chirp(t_template, f0=f0, f1=f1, t1=signal_duration, method='linear')
    errorCount = 0
    for channel in range(channels):
        ch_description = description + '_CH' + str(channel)
        if calculate_distance(template=chirp_template, inputsignal=wave[:,channel], duration_template=signal_duration, samplerate=fs, domain='time', date=today, f0=f0, f1=f1, description=ch_description, display_plots = True, datebased_filename=False) == -1:
            outF.write('Err'+format_nb_digits(errorCount,3) + ': ' + wavname + 'CHANNEL ' + str(channel))
            outF.write('\n')
            errorCount = errorCount + 1
    outF.close()
    
    

### Extra Functions used for easy file management and to ameliorate the recording process ####

def print_countdown(seconds):
    count = seconds
    print('*** Start Countdown ***')
    while count >= 0:
        space = '           '
        if count == 0:
            print(space+'Go!')
        else:
            print(space+str(count))
        time.sleep(1)
        count = count - 1
        
        
def create_date_folder(date):

    date_folder  = './' + str(date.year) + format_nb_digits(date.month,2) + format_nb_digits(date.day,2) + '/'
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
    return date_folder

def get_date_based_filename(date):
    
    # microsecs should always have 6 numbers
    return format_nb_digits(date.hour,2) + format_nb_digits(date.minute,2) + format_nb_digits(date.second,2) + format_nb_digits(date.microsecond,6)
    
def format_nb_digits(number, format_size):
    # format size is amount of digits the number should have, so add zeros if necessary
    # first check how many digits it actually has
    nb_digits = 0
    number_copy = number
    if number == 0:
        zeros = ''
        for zero in range(0,format_size):
            zeros = zeros + '0'
        return zeros
    while number_copy >= 1:
        nb_digits = nb_digits +1
        number_copy = number_copy/10
    nb_zeros = format_size-nb_digits
    if nb_zeros <= 0:
        number = round(number/(10**abs(nb_zeros)))
        return str(number)
    else:
        result = str(number)
        for x in range(0,nb_zeros):
            result = '0'+result
        return result










# Ultimately unused functions that might still be useful for other experiments

# N based
def high_pass(unfiltered, samplerate, fc, N):
    # try fs=32000, fc = 8000, b = 1000; or even good enough with N= 5 :o
    
    # Make sure the window length is odd
    if N%2 == 0:
        N += 1
    
    # Compute sinc filter.
    h = np.sinc(2 * fc / samplerate * (np.arange(N) - (N - 1) / 2))
    
    # Apply window.
    h *= np.blackman(N)
    
    # Normalize to get unity gain.
    h /= np.sum(h)
    
    # Create a high-pass filter from the low-pass filter through spectral inversion.
    h = -h
    h[(N - 1) // 2] += 1
    
    filtered = np.convolve(unfiltered, h)
    return filtered

def gaussian_window(original_signal):
    gaus_window = general_gaussian(51, p=0.5, sig=20)
    windowed = fftconvolve(gaus_window, original_signal)
    windowed = (np.average(original_signal) / np.average(original_signal)) * windowed
    windowed = np.roll(windowed, -25)
    return windowed