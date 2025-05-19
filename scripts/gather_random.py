# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys
import pandas as pd

datadir = '/blue/liu.ying/transcription_bottleneck/data/'

lgs = ['hupa']
task = 'asr'

train_sizes = []

size = 30
while size <= 340:
	if size not in [50, 60, 100, 150, 200, 250]:
		train_sizes.append(size)
	size += 10
method = 'random'

outfile = open(task + '_ave_results_random.txt', 'w')
header = ['Language', 'Task', 'Size', 'Actual_duration', 'Model', 'Metric', 'Value', 'Method'] #, 'Split']
outfile.write(' '.join(tok for tok in header) + '\n')

for size in train_sizes:
	size = str(size)
	for lg in lgs:
		lg = lg.capitalize()
		ave_wer = 0
		ave_cer = 0
		ave_dur = 0
		for seed in range(3):
			seed += 1
			for file in os.listdir(datadir + '/' + lg + '/random/' + size + '/' + str(seed)):
				if file.endswith('eval.txt'):
					for pretrained_model in ['wav2vec2-large-xlsr-53']: #, 'transformer_tiny', 'lstm']:
						evaluation_file = datadir + lg + '/random/' + size + '/' + str(seed) + '/' + pretrained_model + '_test_eval.txt'
						if os.path.exists(evaluation_file):
					#		train_file = pd.read_csv(datadir + lg + '/' + size + '/' + method + '/' + select_interval + '/' + split + '/select' + str(select) + '/' + pretrained_model + '_train.' + size + '.input')
							train_file = pd.read_csv(datadir + lg + '/train_' + size + '_' + str(seed) + '.csv')
							actual_duration = sum([float(dur) for dur in train_file['duration'].tolist()])
							actual_duration = round(actual_duration / 60, 2)
							wer = ''
							cer = ''
							with open(evaluation_file) as f:
								for line in f:
									toks = line.strip().split()
									if line.startswith('WER'):
										wer = toks[1]
									if line.startswith('CER'):
										cer = toks[1]
							ave_wer += float(wer)
							ave_cer += float(cer)
							ave_dur += actual_duration

						else:
							print(evaluation_file)

		info = [lg, task, size, ave_dur/3, pretrained_model, 'wer', ave_wer/3, method]
		outfile.write(' '.join(str(tok) for tok in info) + '\n')

		info = [lg, task, size, ave_dur/3, pretrained_model, 'cer', ave_cer/3, method]
		outfile.write(' '.join(str(tok) for tok in info) + '\n')

