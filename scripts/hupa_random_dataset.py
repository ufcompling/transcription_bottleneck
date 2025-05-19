## Given an input path, in which there is a wav folder and a txt folder
## Randomly partition the audio and their corresponding transcripts into a training and test set

## e.g., python scripts/random_partition.py hupa audio/ data/

import os, re, io
import soundfile as sf
import random
import itertools
import numpy as np
from tqdm import tqdm
from inspect import getouterframes, currentframe
import pandas as pd
import sys

if not os.path.exists('data/'):
	os.system('mkdir data/')

if not os.path.exists('descriptive/'):
	os.system('mkdir descriptive/')

lg = sys.argv[1]
lg = lg.capitalize()

if not os.path.exists('data/' + lg):
	os.system('mkdir data/' + lg)

data_path = 'audio/' + lg + '/'
output_path = 'data/' + lg + '/'
n_random_splits = 3

wav_path = data_path + 'wav/'
transcript_path = data_path + 'txt/'

data = []
total_dur = 0

original_train_data = pd.read_csv(output_path + 'train.csv')
shuffle_train_data = original_train_data.sample(frac=1)

train_path_list = shuffle_train_data['path']
train_transcript_list = shuffle_train_data['transcript']
train_dur_list = shuffle_train_data['duration']

for i in range(len(shuffle_train_data)):
	data.append([train_path_list[i], train_transcript_list[i], train_dur_list[i]])

train_sizes = []

size = 30
while size <= 340:
	if size not in [50, 100, 150, 200, 250]:
		train_sizes.append(size)
	size += 10

for size in train_sizes:
	size *= 60
	ave_train_dur = 0
	for i in range(0, n_random_splits):
		i += 1
		i = str(i)

		train_dur = 0
		train_data = []

		for tok in data:
			if train_dur <= size:
				train_data.append(tok)
				train_dur += tok[-1]

		train_dur = train_dur / 60
		ave_train_dur += train_dur

		train_wav_path_list = [tok[0] for tok in train_data]
		train_transcript_list = [tok[1] for tok in train_data]
		train_dur_list = [tok[-1] for tok in train_data]
		train_speaker_list = ['verdena'] * len(train_dur_list)
		train_output = pd.DataFrame({'path': train_wav_path_list,
			  	'transcript': train_transcript_list,
			  	'duration': train_dur_list,
			  	'speaker': train_speaker_list})

		train_output.to_csv(output_path + 'train_' + str(int(size / 60)) + '_' + str(i) + '.csv', index = False)		

	print(size, ave_train_dur / 3)
