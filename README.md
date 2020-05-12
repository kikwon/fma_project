# FMA Project: Genre Classification and Popularity Regression

## Intro
This project will be an application-oriented project with two potential tasks.  One will be to see how well we can classify the genre of a song given the audio file.  The second task will be a prediction of the popularity of a song using a regression based on the audio files.

## Links
data source: http://archive.ics.uci.edu/ml/datasets/FMA%3A+A+Dataset+For+Music+Analysis#

FMA dataset Research Paper: https://arxiv.org/pdf/1612.01840.pdf

Proposal: https://github.com/kikwon/fma_project/blob/master/Proposal.pdf

### Script extract_mfcc.py guide:  
The mfcc features that come with features.csv are different from what we get when we extract the mfcc from audio directly. This script will extract mfcc features (1 form of spectrogram) for some window of the audio (1-2 minutes). It will output a pickled list of mfcc features ('mfcc.pkl') in the same directory. You can import this pickle later in notebook. Some code to process to get it ready for the neural network training is included in 'Basic_arch_mfcc_from_features_csv.ipynb' in the 'from audio' section. This script runs for 2 hours. I used discrete cosine transform type 3 for more contrast (read up Librosa's documentations for further info).

Libraries needed:
Librosa  
Other standard python modules

Instructions:  
In the command prompt or terminal type:
python3 extract_mfcc.py 'path_to_fma_small_folder'  
For me it was: python3 extract_mfcc.py /Downloads/fma_small  
Feel free to modify the script to extract any features you want.
