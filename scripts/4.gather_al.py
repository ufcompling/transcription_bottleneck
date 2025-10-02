# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys
import pandas as pd

<<<<<<< HEAD
datadir = '/blue/liu.ying/transcription_bottleneck/data/'
=======
datadir = '/orange/ufdatastudios/zoey.liu/transcription_bottleneck/data/'
>>>>>>> 644c7d9e26d72a5fe24f6887da0745ffc880f538

select_interval = '15' 
task = 'asr'
sizes = ['15']
#split = sys.argv[1]

#outfile = open(task + '_results_' + split + '.txt', 'w')


for size in sizes:
	outfile = open(task + '_results_' + size + '.txt', 'w')
	header = ['Language', 'Task', 'Size', 'Select_interval', 'Select_size', 'Actual_duration', 'Model', 'Metric', 'Value', 'Method'] #, 'Split']
	outfile.write(' '.join(tok for tok in header) + '\n')

<<<<<<< HEAD
	for lg in os.listdir('/blue/liu.ying/transcription_bottleneck/pbs/'):
=======
	for lg in os.listdir('/orange/ufdatastudios/zoey.liu/transcription_bottleneck/data/'):
>>>>>>> 644c7d9e26d72a5fe24f6887da0745ffc880f538
		lg = lg.split('_')[0]
		descriptive = pd.read_csv('descriptive/' + lg + '.txt', sep = '\t')
		full_train_dur = descriptive['train duration'].tolist()[0]
		print(lg, full_train_dur)
		max_size = ''
		if 'h' in full_train_dur:
			full_train_dur = full_train_dur.split('h')
			max_size = int(full_train_dur[0]) * 60 + int(full_train_dur[1][ : -3])
		else:
			max_size = int(full_train_dur[ : -3])
		
		if size in os.listdir(datadir + lg):
			for pretrained_model in ['wav2vec2-large-xlsr-53']:
				for method in ['al']:
					for seed in ['1']:
						iterations = int(max_size / int(select_interval))
						for i in range(iterations):
							select = str(i * int(select_interval))
							evaluation_file = datadir + lg + '/' + size + '/' + method + '/' + seed + '/' + select_interval + '/select' + str(select) + '/' + pretrained_model + '_test_eval.txt'
							if os.path.exists(evaluation_file):
								train_file = pd.read_csv(datadir + lg + '/' + size + '/' + method + '/' + seed + '/' + select_interval + '/select' + str(select) + '/' + pretrained_model + '_train.' + size + '.input')
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

								info = [lg, task, size, select_interval, select, actual_duration, pretrained_model, 'wer', wer, method]
								outfile.write(' '.join(str(tok) for tok in info) + '\n')
								info = [lg, task, size, select_interval, select, actual_duration, pretrained_model, 'cer', cer, method]
								outfile.write(' '.join(str(tok) for tok in info) + '\n')

							else:
								print(lg, select)

select = 'all'
#evaluation_file = datadir + lg + '/' + size + '/' + method + '/' + select_interval + '/' + split + '/select' + str(select) + '/' + pretrained_model + '_test_eval.txt'
evaluation_file = datadir + lg + '/' + size + '/' + method + '/' + select_interval + 'select' + str(select) + '/' + pretrained_model + '_test_eval.txt'
if os.path.exists(evaluation_file):
	wer = ''
	cer = ''
	with open(evaluation_file) as f:
		for line in f:
			toks = line.strip().split()
			if line.startswith('WER'):
				wer = toks[1]
			if line.startswith('CER'):
				cer = toks[1]

	info = [lg, task, size, select_interval, select, max_size, pretrained_model, 'wer', wer, 'ALL']
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [lg, task, size, select_interval, select, max_size, pretrained_model, 'cer', cer, 'ALL']
	outfile.write(' '.join(str(tok) for tok in info) + '\n')

