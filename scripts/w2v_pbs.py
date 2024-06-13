import io, os, sys
import soundfile as sf
import pandas as pd

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

lg = sys.argv[1]
print(lg)
size = sys.argv[2]
select_interval = sys.argv[3]

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

pwd; hostname; date

module load conda
mamba activate whisper

cd /blue/liu.ying/transcription_bottleneck/

'''

descriptive = pd.read_csv('descriptive/' + lg + '.txt', sep = '\t')
full_train_dur = descriptive['train duration'].tolist()[0]
full_train_dur = full_train_dur.split('h')
max_size = int(full_train_dur[0]) * 60 + int(full_train_dur[1][ : -3])

pretrained_model = "wav2vec2-large-xlsr-53" #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b
method = 'al'

iterations = int(max_size / int(select_interval))

with open('pbs/' + lg + '_' + size + '_' + select_interval + '_' + method + '.pbs', 'w') as f:
	first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + lg + '_' + size + '_' + select_interval + '    # Job name'

	f.write(first_string + '\n')
	f.write(second_string + '\n')

	for i in range(iterations):
		select = str(i * int(select_interval))
		if int(size) + int(select) <= max_size: # and int(select) >= 10000:
						
			f.write('python scripts/train_w2v.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --pretrained_model ' + pretrained_model + '\n')
			f.write('\n')
			
			f.write('python scripts/eval_w2v_confidence.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --pretrained_model ' + pretrained_model + '\n')			
			f.write('\n')

	select = 'all'

	f.write('python scripts/train_w2v.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --pretrained_model ' + pretrained_model + '\n')
	f.write('\n')
			
	f.write('python scripts/eval_w2v_confidence.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --pretrained_model ' + pretrained_model + '\n')			
	f.write('\n')
			
	f.write('date' + '\n')
	f.write('\n')

'''
### Combine pbs file for each lguage, given a start size, a select interval, and a select size

iterations = int(max_size / int(select_interval))
for treebank in treebanks_select:
	for method in ['al', 'tokenfreq']:
		for size in sizes:
			together_file = open('pbs/' + treebank + '_' + size + '_' + select_interval +  '_' + method + '.sh', 'w') # doing sbatch all together
			c = 0
			for i in range(iterations):
				select = str(i * int(select_interval))
				for arch in ['transformer']:
					evaluation_file = '/blue/liu.ying/unlabeled_pos/' + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch + '/eval.txt'
					if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size != 0:
						pass
					else:
						together_file.write('sbatch pbs/' + treebank + size + '_select' + select + '_' + arch + '_' + method + '.pbs' + '\n')
						c += 1

			if c == 0: ## deleting empty together_file
				os.system('rm ' + 'pbs/' + treebank + '_' + size + '_' + select_interval +  '_' + method + '.sh')

'''

'''
## For each start size, combine pbs files for all lguages for a given select interval and select size
iterations = int(max_size / int(select_interval))
for i in range(iterations):
	select = str(i * int(select_interval))
	for size in sizes:
		together_file = open('pbs_new/' + task + size + '_' + select_interval + '_select' + select + '.sh', 'w') # doing sbatch all together
		c = 0
		for lg in lgs:			
			for arch in ['transformer']:
				if lg + '_' + task + size + '_' + select_interval + '_select' + select + '_' + arch +'.pbs' in os.listdir('pbs_new/'):
					evaluation_file = '/blue/liu.ying/al_morphseg/al_trainselect/' + lg + '_' + task + size + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
					if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size != 0:
						pass
					else:
						together_file.write('sbatch pbs_new/' + lg + '_' + task + size + '_' + select_interval + '_select' + select + '_' + arch +'.pbs' + '\n')
						c += 1
		if c == 0:
			os.system('rm ' + 'pbs_new/' + task + size + '_' + select_interval + '_select' + select + '.sh')
'''