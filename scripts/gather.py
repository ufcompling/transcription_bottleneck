# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys

datadir = '/blue/liu.ying/transcription_bottleneck/data/'

lgs = ['hupa']

select_interval = '5' 

sizes = ['30']

task = 'asr'
outfile = open(task + '_results.txt', 'w')

descriptive = pd.read_csv('descriptive/' + lg + '.txt', sep = '\t')
full_train_dur = descriptive['train duration'].tolist()[0]
full_train_dur = full_train_dur.split('h')
max_size = int(full_train_dur[0]) * 60 + int(full_train_dur[1][ : -3])

header = ['Language', 'Size', 'Select_interval', 'Select_size', 'Model', 'Metric', 'Value']
outfile.write(' '.join(tok for tok in header) + '\n')

for size in sizes:
	for lg in lgs:
		lg = lg.capitalize()
		if size in os.listdir(datadir + lg):
			for pretrained_model in ['wav2vec2-large-xlsr-53']: #, 'transformer_tiny', 'lstm']:
				for method in ['al']:
					iterations = int(max_size / int(select_interval))
					for i in range(iterations):
						select = str(i * int(select_interval))
						evaluation_file = datadir + lg + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + pretrained_model + '_test_eval.txt'
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

							info = [lg, task, size, select_interval, select, pretrained_model, 'wer', wer]
							outfile.write(' '.join(str(tok) for tok in info) + '\n')
							info = [lg, task, size, select_interval, select, pretrained_model, 'cer', cer]
							outfile.write(' '.join(str(tok) for tok in info) + '\n')

						select += int(select_interval)

select = 'all'
evaluation_file = datadir + lg + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + pretrained_model + '_test_eval.txt'
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

	info = [lg, task, size, select_interval, select, pretrained_model, 'wer', wer]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [lg, task, size, select_interval, select, pretrained_model, 'cer', cer]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
