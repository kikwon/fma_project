#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
from tqdm import tqdm
import librosa
import warnings
import sys
import pickle
import numpy as np

def mp3gen(path):
    for root, dirs, files in os.walk(path):
        for filename in files:
            if os.path.splitext(filename)[1] == ".mp3":
                yield os.path.join(root, filename)
                
def fxn():
    warnings.warn('User warning', UserWarning)
    
def main(path):
    filenames = []
    for mp3file in mp3gen(path):
        filenames.append(mp3file)
    
    filenames = sorted(filenames, key=lambda x: (x.split('.')[0]))
    mel_spec_data = []
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fxn()
        for mp3 in tqdm(filenames):
            try:
                data, sampling_rate = librosa.load(mp3, mono=True)
                mel_spec = librosa.feature.melspectrogram(y=data, sr=sampling_rate, n_fft=2048, hop_length=512)
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_data.append(mel_spec)
            except Exception as e:
                print(e)
                mel_spec_data.append(np.nan)
                
    return mel_spec_data


if __name__ == "__main__":
    mel_spec = main(sys.argv[1])
    
    with open('mel-spec.pkl', 'wb') as fp:
        pickle.dump(mel_spec, fp)

