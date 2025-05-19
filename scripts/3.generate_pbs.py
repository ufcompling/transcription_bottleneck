# e.g., python3 scripts/w2v_pbs.py Hupa 30 5

import io, os, sys
import soundfile as sf
import pandas as pd

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

lg = sys.argv[1]
if lg != 'isiXhosa':
	lg = lg.capitalize()

size = '60' #sys.argv[2]
select_interval = '30' #sys.argv[3]

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios

pwd; hostname; date

whisper_path=/blue/liu.ying/conda/envs/whisper/bin
export PATH=$whisper_path:$PATH

cd /orange/ufdatastudios/zoey.liu/transcription_bottleneck/

'''

descriptive = pd.read_csv('descriptive/' + lg + '.txt', sep = '\t')
full_train_dur = descriptive['train duration'].tolist()[0]
max_size = ''
if 'h' in full_train_dur:
	full_train_dur = full_train_dur.split('h')
	max_size = int(full_train_dur[0]) * 60 + int(full_train_dur[1][ : -3])
else:
	max_size = int(full_train_dur[ : -3])

pretrained_model = "wav2vec2-xls-r-2b" #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b, wav2vec2-large-xlsr-53

method = sys.argv[2]

iterations = int(max_size / int(select_interval))

seed = '1'

with open('pbs/' + lg + '_' + size + '_' + select_interval + '_' + method + '.pbs', 'w') as f:
	first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + lg + '_' + size + '_' + select_interval + '    # Job name'

	f.write(first_string + '\n')
	f.write(second_string + '\n')

	for i in range(iterations):
		select = str(i * int(select_interval))
		if int(size) + int(select) <= max_size: # and int(select) >= 10000:
						
			f.write('python scripts/train_' + method + '_w2v.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --seed ' + seed + ' --pretrained_model ' + pretrained_model + '\n')
			f.write('\n')
			f.write('python scripts/eval_' + method + '_w2v_individual.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --seed ' + seed + ' --pretrained_model ' + pretrained_model + '\n')
			f.write('\n')
						
			f.write('\n')

#	if method == 'al':
#		select = 'all'

#		f.write('python scripts/train_al_w2v.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --seed ' + seed +  ' --pretrained_model ' + pretrained_model + '\n')
#		f.write('\n')
			
#		f.write('python scripts/eval_al_w2v.py --lang ' + lg + ' --size ' + size + ' --interval ' + select_interval + ' --select ' + select + ' --method ' + method + ' --seed ' + seed +  ' --pretrained_model ' + pretrained_model + '\n')			
#		f.write('\n')
			
#		f.write('date' + '\n')
#		f.write('\n')

