
# Based on
# https://github.com/ShawnHymel/tflite-speech-recognition

import sounddevice as sd
import numpy as np
import scipy.signal
import timeit
import python_speech_features
import pygame
from tflite_runtime.interpreter import Interpreter
import smtp 
word_threshold = 0.7
rec_duration = 0.5
window_stride = 0.5
sample_rate = 48000
resample_rate = 16000
num_channels = 1
model_path = 'model_sinc.tflite'
flag=0
pygame.init()

my_sound = pygame.mixer.Sound('Intruder-alert.wav')
#keywords
word2index = {
    # core words
    "backward": 0,
    "bed": 1,
    "bird": 2,
    "cat": 3,
    "dog": 4,
    "down": 5,
    "eight": 6,
    "five": 7,
    "follow": 8,
    "forward": 9,
    "four": 10,
    "go": 11,
    "happy": 12,
    "house": 13,
    "learn": 14,
    "left": 15,
    "marvin": 16,
    "nine": 17,
    "no": 18,
    "off": 19,
    "on":20,
    "one":21,
    "right":22,
    "seven":23,
    "sheila":24,
    "six":25,
    "stop":26,
    "three":27,
    "tree":28,
    "two":29,
    "up":30,
    "visual":31,
    "wow":32,
    "yes":33,
    "zero":34
}

index2word = [word for word in word2index]
# Sliding window
window = np.zeros(int(rec_duration * resample_rate) * 2)


# Load model (interpreter)
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)

# Decimate (filter and downsample)
def decimate(signal, old_fs, new_fs):
    
    # Check to make sure we're downsampling
    if new_fs > old_fs:
        print("Error: target sample rate higher than original")
        return signal, old_fs
    
    # We can only downsample by an integer factor
    dec_factor = old_fs / new_fs
    if not dec_factor.is_integer():
        print("Error: can only decimate by integer factor")
        return signal, old_fs

    # Do decimation
    resampled_signal = scipy.signal.decimate(signal, int(dec_factor))

    return resampled_signal, new_fs

# This gets called every 0.5 seconds
def sd_callback(rec, frames, time, status):

    global flag
    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    # Convert from stereo (2 dimensions) to mono
    rec = np.squeeze(rec)
    
    # Resample
    rec, new_fs = decimate(rec, sample_rate, resample_rate)
    
    # Save recording onto sliding window
    window[:len(window)//2] = window[len(window)//2:]
    window[len(window)//2:] = rec

    # Compute features - ensure same set up as is used in 
    #training
    print(window.shape)

    # Make prediction from model
    in_tensor = np.float32(window.reshape(1, 1, window.shape[0]))
    interpreter.set_tensor(input_details[0]['index'], in_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data.squeeze(0)
    #print(output_data)
    prediction_sorted_indices = output_data.argsort()
    prediction = output_data.argmax()
    print(prediction)
    print("candidates:\n-----------------------------")
    for k in range(3):
       i = int(prediction_sorted_indices[-1-k])
       print("i",i)
       #print("%d.)\t%s\t:\t%2.1f%%" % (k+1, index2word[i], output_data[i]*100))
    
    if(prediction == 13 and output_data[prediction] > word_threshold):
        
        my_sound.play()
        flag=1
  

    
    if(prediction == 5 and output_data[prediction] > word_threshold):
        
        smtp.sendemail()
        flag=1

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while flag==0:
        pass

