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

for i in range(1, n_random_splits):
	i += 1
	i = str(i)
	random.shuffle(data)

	train_dur = 0
	test_dur = 0
	train_data = []
	test_data = []

	for tok in data:
		if train_dur <= total_dur * 0.8:
			train_data.append(tok)
			train_dur += tok[-1]
		else:
			test_data.append(tok)
			test_dur += tok[-1]

	print(n_random_splits)
	train_dur = train_dur / 3600
	test_dur = test_dur / 3600
	train_h = int(train_dur)
	train_min = int((train_dur - train_h) * 60)
	test_h = int(test_dur)
	test_min = int((test_dur - test_h) * 60)

	with open('descriptive/' + lg + '.txt', 'w') as f:
		f.write('train duration\ttest duration' + '\n')
		f.write(str(train_h) + 'h' + str(train_min) + 'min\t' + str(test_h) + 'h' + str(test_min) + 'min' + '\n')
	print('train duration: ' + str(train_h) + 'h' + str(train_min) + 'min')
	print('test duration: ' + str(test_h) + 'h' + str(test_min) + 'min')
	print('')

	train_wav_path_list = [wav_path + tok[0] for tok in train_data]
	train_transcript_list = [tok[1] for tok in train_data]
	train_dur_list = [tok[-1] for tok in train_data]
	if lg == 'Hupa':
		train_speaker_list = ['verdena'] * len(train_dur_list)
	train_output = pd.DataFrame({'path': train_wav_path_list,
			  	'transcript': train_transcript_list,
			  	'duration': train_dur_list,
			  	'speaker': train_speaker_list})

	train_output.to_csv(output_path + 'train_' + str(i) + '.csv', index = False)

	test_wav_path_list = [wav_path + tok[0] for tok in test_data]
	test_transcript_list = [tok[1] for tok in test_data]
	test_dur_list = [tok[-1] for tok in test_data]
	if lg == 'Hupa':
		test_speaker_list = ['verdena'] * len(test_dur_list)
	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list,
			  	'duration': test_dur_list,
			  	'speaker': test_speaker_list})

	test_output.to_csv(output_path + 'test_' + str(i) + '.csv', index = False)

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

	train_output.to_csv(output_path + 'train_4.csv', index = False)

	test_wav_path_list = [wav_path + tok[0] for tok in top_data]
	test_transcript_list = [tok[1] for tok in top_data]
	test_dur_list = [tok[-1] for tok in top_data]
	if lg == 'Hupa':
		test_speaker_list = ['verdena'] * len(test_dur_list)
	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list,
			  	'duration': test_dur_list,
			  	'speaker': test_speaker_list})

	test_output.to_csv(output_path + 'test_4.csv', index = False)

top_dur = top_dur / 3600
second_dur = second_dur / 3600
top_h = int(top_dur)
top_min = int((top_dur - top_h) * 60)
second_h = int(second_dur)
second_min = int((second_dur - second_h) * 60)
print('top duration: ' + str(top_h) + 'h' + str(top_min) + 'min')
print('second duration: ' + str(second_h) + 'h' + str(second_min) + 'min')
print('')

'''
for split in ['train', 'test']: 
	for file in os.listdir(data_path + split + '/'):
		if file.endswith('.wav'):
			wav_path = data_path + split + '/' + file
			file_name = file.split('.')[0]
			transcipt_file = data_path + split + '/' + file_name + '.txt'
			transcript = ''
			with open(transcipt_file) as f:
				for line in f:
					transcript = line.strip()

			wav_path_list.append(wav_path)
			transcript_list.append(transcript)

	output_data = pd.DataFrame({'path_to_the_wav_file': wav_path_list,
			  	'transcript': transcript_list})

	output_data.to_csv(data_path + split + '.csv', index = False)
'''