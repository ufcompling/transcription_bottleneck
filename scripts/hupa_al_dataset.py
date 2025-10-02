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

wav_path = data_path + 'wav/'
transcript_path = data_path + 'txt/'

data = []
total_dur = 0

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
			total_dur += dur
			data.append([file, transcript, dur])
		else:
			pass

### Now outputing data with the top tier as the test set and the second tier as the training set

top_data = []
top_dur = 0
second_data = []
second_dur = 0
for tok in data:
	file = tok[0]
	if '_1_' in file:
		print(file)
		top_data.append(tok)
		top_dur += tok[-1]
		print(top_dur, tok[-1])
	else:
		second_data.append(tok)
		second_dur += tok[-1]

	train_wav_path_list = [wav_path + tok[0] for tok in second_data]
	train_transcript_list = [tok[1] for tok in second_data]
	train_dur_list = [tok[-1] for tok in second_data]
	if lg == 'Hupa':
		train_speaker_list = ['verdena'] * len(train_dur_list)
	train_output = pd.DataFrame({'path': train_wav_path_list,
			  	'transcript': train_transcript_list,
			  	'duration': train_dur_list,
			  	'speaker': train_speaker_list})

	train_output.to_csv(output_path + 'train.csv', index = False)

	test_wav_path_list = [wav_path + tok[0] for tok in top_data]
	test_transcript_list = [tok[1] for tok in top_data]
	test_dur_list = [tok[-1] for tok in top_data]
	if lg == 'Hupa':
		test_speaker_list = ['verdena'] * len(test_dur_list)
	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list,
			  	'duration': test_dur_list,
			  	'speaker': test_speaker_list})

	test_output.to_csv(output_path + 'test.csv', index = False)

top_dur = top_dur / 3600
second_dur = second_dur / 3600
top_h = int(top_dur)
top_min = int((top_dur - top_h) * 60)
second_h = int(second_dur)
second_min = int((second_dur - second_h) * 60)
print('top duration: ' + str(top_h) + 'h' + str(top_min) + 'min')
print('second duration: ' + str(second_h) + 'h' + str(second_min) + 'min')
print('')

