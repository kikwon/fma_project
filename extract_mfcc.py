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
    mfcc_data = []
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fxn()
        for mp3 in tqdm(filenames):
            try:
                data, sampling_rate = librosa.load(mp3, mono=True)
                mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate, dct_type=3)
                mfcc_data.append(mfcc)
            except Exception as e:
                print(e)
                mfcc_data.append(np.nan)
                
    return mfcc_data

#if __name__ == "__main__":
    
    #main(sys.argv)      


# In[ ]:


if __name__ == "__main__":
    mfcc = main(sys.argv[1])
    
    with open('mfcc.pkl', 'wb') as fp:
        pickle.dump(mfcc, fp)

