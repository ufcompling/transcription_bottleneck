## Given an input path, in which there is a wav folder and a txt folder
## Randomly partition the audio and their corresponding transcripts into a training and test set

## e.g., python scripts/random_partition.py hupa

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

# Store descriptive statistics for each language
if not os.path.exists('descriptive/'):
	os.system('mkdir descriptive/')

lg = sys.argv[1]
if lg != 'isiXhosa':
	lg = lg.capitalize()

# Store output csv files for each language
if not os.path.exists('data/' + lg):
	os.system('mkdir data/' + lg)

data_path = '/blue/liu.ying/asr_corpora/' + lg + '/'
output_path = 'data/' + lg + '/'
n_random_splits = 2

wav_path = data_path + 'wav/'
transcript_path = data_path + 'txt/'

data = []
total_dur = 0

for file in os.listdir(wav_path):
	if file.endswith('wav') or file.endswith('flac'):
		if total_dur <= 100 * 3600: # 100 hours
			file_name = file.split('.')[0]
			sr = 16000
			try:
				signal, sr = sf.read(wav_path + file) # signal and sampling rate
				dur = len(signal) / sr # audio duration		

				# For convenience, we have the same file name for wav and transcript
				# except for different file extensions of course
				transcript_file = file_name + '.txt'		
				if os.path.exists(transcript_path + transcript_file):
					transcript = ''
					with open(transcript_path + transcript_file) as f:
						for line in f:
							transcript = line.strip()

					# Only including audio that is at least 5s
					# wav2vec does not handle short audio very well
					if dur >= 5 and len(transcript.split()) > 1:
						total_dur += dur
						data.append([file, transcript, dur])
					else:
						pass
				else:
					print('Transcript does not exist for ' + file)
			except:
				print('Error reading ' + file)
		else:
			break

print('Total duration: ' + str(total_dur / 3600) + ' hours')

for i in range(0, n_random_splits):
	i += 1
	i = str(i)
	random.shuffle(data)

	train_dur = 0
	test_dur = 0
	train_data = []
	test_data = []

	for tok in data:

		## Splitting the full corpus into training and test at a 4:1 ratio
		if train_dur <= total_dur * 0.8:
			train_data.append(tok)
			train_dur += tok[-1]
		else:
			test_data.append(tok)
			test_dur += tok[-1]

	# Up until this point, train_dur is in seconds
	# Converting to hours
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

	## TODO: for other languages, we want to check if speaker information is available
	else:
		train_speaker_list = ['empty'] * len(train_dur_list)

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
	## TODO: for other languages, we want to check if speaker information is available
	else:
		test_speaker_list = ['empty'] * len(test_dur_list)

	test_output = pd.DataFrame({'path': test_wav_path_list,
			  	'transcript': test_transcript_list,
			  	'duration': test_dur_list,
			  	'speaker': test_speaker_list})

	test_output.to_csv(output_path + 'test_' + str(i) + '.csv', index = False)



