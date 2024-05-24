## Given an input path, in which there is a wav folder and a txt folder
## Randomly partition the audio and their corresponding transcripts into a training and test set

import os, re, io
import soundfile as sf
import random
import itertools
import numpy as np
from tqdm import tqdm
from inspect import getouterframes, currentframe
import pandas as pd
import sys

data_path = sys.argv[1]
output_path = sys.argv[2]
n_random_splits = 1

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
		total_dur += dur

		transcript_file = file_name + '.txt'
		transcript = ''
		with open(transcript_path + transcript_file) as f:
			for line in f:
				transcript = line.strip()
		data.append([file, transcript, dur])

for i in range(n_random_splits):
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

	print('train duration: ' + str(train_h) + 'h' + str(train_min) + 'min')
	print('test duration: ' + str(test_dur) + 'h' + str(test_min) + 'min')
	print('')

	train_wav_path_list = [wav_path + tok[0] for tok in train_data]
	train_transcript_list = [tok[1] for tok in train_data]
	train_output = pd.DataFrame({'path': train_wav_path_list,
			  	'transcript': train_transcript_list})

	train_output.to_csv(output_path + 'train.csv', index = False)

	test_wav_path_list = [wav_path + tok[0] for tok in test_data]
	test_transcript_list = [tok[1] for tok in test_data]
	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list})

	test_output.to_csv(output_path + 'test.csv', index = False)


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