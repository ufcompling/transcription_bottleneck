## Given an input path, in which there is a wav folder and a txt folder
## Iteratively construct randomly sampled training sets of different sizes
## e.g., python scripts/random_dataset.py isiXhosa

import os, re, io
import soundfile as sf
import random
import itertools
import numpy as np
from tqdm import tqdm
from inspect import getouterframes, currentframe
import pandas as pd
import sys

lg = sys.argv[1]
if lg != 'isiXhosa':
	lg = lg.capitalize()

size = '60' #sys.argv[2]
select_interval = '30' #sys.argv[3]

descriptive = pd.read_csv('descriptive/' + lg + '.txt', sep = '\t')
full_train_dur = descriptive['train duration'].tolist()[0]
max_size = ''
if 'h' in full_train_dur:
	full_train_dur = full_train_dur.split('h')
	max_size = int(full_train_dur[0]) * 60 + int(full_train_dur[1][ : -3])
else:
	max_size = int(full_train_dur[ : -3])

<<<<<<< HEAD
pretrained_model = "wav2vec2-xls-r-2b"
=======
pretrained_model = "wav2vec2-large-xlsr-53" #"wav2vec2-xls-r-2b"
>>>>>>> 644c7d9e26d72a5fe24f6887da0745ffc880f538

data_path = '/orange/ufdatastudios/zoey.liu/transcription_bottleneck/data/' + lg + '/'
n_random_splits = 1
seed = '1'

if not os.path.exists(data_path + size):
	os.system('mkdir ' + data_path + size)

if not os.path.exists(data_path + size + '/random'):
	os.system('mkdir ' + data_path + size + '/random')

if not os.path.exists(data_path + size + '/random/' + seed):
	os.system('mkdir ' + data_path + size + '/random/' + seed)

if not os.path.exists(data_path + size + '/random/' + seed + '/' + select_interval):
	os.system('mkdir ' + data_path + size + '/random/' + seed + '/' + select_interval)

wav_path = data_path + 'wav/'
transcript_path = data_path + 'txt/'

original_train_data = pd.read_csv(data_path + 'train_' + seed + '.csv')
shuffle_train_data = original_train_data.sample(frac=1)

train_path_list = shuffle_train_data['path']
train_transcript_list = shuffle_train_data['transcript']
train_dur_list = shuffle_train_data['duration']
train_speaker_list = shuffle_train_data['speaker']

size_list = []
iterations = int(max_size / int(select_interval))
for i in range(iterations):
	select = str(i * int(select_interval))
	if int(size) + int(select) <= max_size:
		size_list.append(int(size) + int(select))
		if not os.path.exists(data_path + size + '/random/' + seed + '/' + select_interval + '/select' + select):
			os.system('mkdir ' + data_path + size + '/random/' + seed + '/' + select_interval + '/select' + select)

print(size_list[0], size_list[-1])

for tok in size_list:
	sampled_train_path_list = []
	sampled_train_transcript_list = []
	sampled_train_dur_list = []
	sampled_train_speaker_list = []
	sampled_train_dur = 0

	sampled_select_path_list = []
	sampled_select_transcript_list = []
	sampled_select_dur_list = []
	sampled_select_speaker_list = []

	for i in range(len(shuffle_train_data)):
		if sampled_train_dur < tok * 60:
			sampled_train_path_list.append(train_path_list[i])
			sampled_train_transcript_list.append(train_transcript_list[i]) 
			sampled_train_dur_list.append(train_dur_list[i]) 
			sampled_train_speaker_list.append(train_speaker_list[i])
			sampled_train_dur += float(train_dur_list[i])		
		else:
			train_output = pd.DataFrame({'path': sampled_train_path_list,
			  	'transcript': sampled_train_transcript_list,
			  	'duration': sampled_train_dur_list,
			  	'speaker': sampled_train_speaker_list})

			sampled_select_path_list = train_path_list[i:]
			sampled_select_transcript_list = train_transcript_list[i:]
			sampled_select_dur_list = train_dur_list[i:]
			sampled_select_speaker_list = train_speaker_list[i:]
			select_output = pd.DataFrame({'path': sampled_select_path_list,
			  	'transcript': sampled_select_transcript_list,
			  	'duration': sampled_select_dur_list,
			  	'speaker': sampled_select_speaker_list})
			
			break

#	print(data_path + size + '/random/' + seed + '/' + select_interval + '/select' + str(tok - int(size)) + '/' + pretrained_model + '_train.' + size + '.input')
	train_output.to_csv(data_path + size + '/random/' + seed + '/' + select_interval + '/select' + str(tok - int(size)) + '/' + pretrained_model + '_train.' + size + '.input', index = False)
	select_output.to_csv(data_path + size + '/random/' + seed + '/' + select_interval + '/select' + str(tok - int(size)) + '/' + pretrained_model + '_select.' + size + '.input', index = False)


<<<<<<< HEAD
=======

>>>>>>> 644c7d9e26d72a5fe24f6887da0745ffc880f538
