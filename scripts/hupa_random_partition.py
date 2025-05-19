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

top_data = []
top_total_dur = 0
second_data = []
second_total_dur = 0

for file in os.listdir(wav_path):
	if file.endswith('wav'):
		file_name = file.split('.')[0]
		sr = 16000
		signal, sr = sf.read(wav_path + file) # signal and sampling rate
		dur = len(signal) / sr # audio duration		

		transcript_file = file_name + '.txt'
		transcript = ''
		with open(transcript_path + transcript_file) as f:
			for line in f:
				transcript = line.strip()
		if dur >= 5 and len(transcript.split()) > 1:
			if '_1_' in file:
				top_data.append([file, transcript, dur])
				top_total_dur += dur
			else:
				second_data.append([file, transcript, dur])
				second_total_dur += dur	
		else:
			pass

top_total_dur = top_total_dur / 3600
second_total_dur = second_total_dur / 3600
top_h = int(top_total_dur)
top_min = int((top_total_dur - top_h) * 60)
second_h = int(second_total_dur)
second_min = int((second_total_dur - second_h) * 60)
print('top total duration: ' + str(top_h) + 'h' + str(top_min) + 'min')
print('second total duration: ' + str(second_h) + 'h' + str(second_min) + 'min')
print('')

descriptive_file = open('descriptive/' + lg + '.txt', 'w')
descriptive_file.write('top total duration: ' + str(top_h) + 'h' + str(top_min) + 'min' + '\n')
descriptive_file.write('second total duration: ' + str(second_h) + 'h' + str(second_min) + 'min' + '\n')

### Output second tier data
second_wav_path_list = [wav_path + tok[0] for tok in second_data]
second_transcript_list = [tok[1] for tok in second_data]
second_dur_list = [tok[-1] for tok in second_data]
if lg == 'Hupa':
	second_speaker_list = ['verdena'] * len(second_dur_list)
second_output = pd.DataFrame({'path': second_wav_path_list,
			'transcript': second_transcript_list,
			'duration': second_dur_list,
			'speaker': second_speaker_list})

second_output.to_csv(output_path + 'second_tier.csv', index = False)

### Constructing training and test sets
### 3 random splits of the top tier data, 4:1 ratio
### thereby 3 test sets
### For each test set
### Scenario 1: training data only comes from top tier
### Scenario 2: already have all annotations in top tier, trying to collect additional annotations from second tier
### Scenario 3: combine top and second tier data together, mimicking the scenario where there are multiple annotators with dfferent levels of expertise

for i in range(n_random_splits):
	i += 1
	i = str(i)
	random.shuffle(top_data)

	top_train_dur = 0
	top_test_dur = 0
	top_train_data = []
	top_test_data = []

	for tok in top_data:
		if top_train_dur <= top_total_dur * 3600 * 0.8:
			top_train_data.append(tok)
			top_train_dur += tok[-1]
		else:
			top_test_data.append(tok)
			top_test_dur += tok[-1]

	top_train_dur = top_train_dur / 3600
	top_test_dur = top_test_dur / 3600
	top_train_h = int(top_train_dur)
	top_train_min = int((top_train_dur - top_train_h) * 60)
	top_test_h = int(top_test_dur)
	top_test_min = int((top_test_dur - top_test_h) * 60)

	descriptive_file.write('top train duration ' + i + ': ' + str(top_train_h) + 'h' + str(top_train_min) + 'min' + '\n')
	descriptive_file.write('top test duration ' + i + ': ' + str(top_test_h) + 'h' + str(top_test_min) + 'min' + '\n')
	print('top train duration ' + i + ': ' + str(top_train_h) + 'h' + str(top_train_min) + 'min')
	print('top test duration ' + i + ': ' + str(top_test_h) + 'h' + str(top_test_min) + 'min')
	print('')

	### Construct training data from top tier
	top_train_wav_path_list = [wav_path + tok[0] for tok in top_train_data]
	top_train_transcript_list = [tok[1] for tok in top_train_data]
	top_train_dur_list = [tok[-1] for tok in top_train_data]
	if lg == 'Hupa':
		top_train_speaker_list = ['verdena'] * len(top_train_dur_list)
	top_train_output = pd.DataFrame({'path': top_train_wav_path_list,
				'transcript': top_train_transcript_list,
				'duration': top_train_dur_list,
				'speaker': top_train_speaker_list})

	top_train_output.to_csv(output_path + 'top_train_' + str(i) + '.csv', index = False)

	### Construct test data from top tier
	top_test_wav_path_list = [wav_path + tok[0] for tok in top_test_data]
	top_test_transcript_list = [tok[1] for tok in top_test_data]
	top_test_dur_list = [tok[-1] for tok in top_test_data]
	if lg == 'Hupa':
		top_test_speaker_list = ['verdena'] * len(top_test_dur_list)
	top_test_output = pd.DataFrame({'path': top_test_wav_path_list,
				'transcript': top_test_transcript_list,
				'duration': top_test_dur_list,
				'speaker': top_test_speaker_list})

	top_test_output.to_csv(output_path + 'top_test_' + str(i) + '.csv', index = False)

