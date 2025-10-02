# e.g., python3 scripts/w2v_pbs.py Hupa 30 5

import io, os, sys
import soundfile as sf
import pandas as pd

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

lg = sys.argv[1]
print(lg)

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=7gb                     # Job memory request
#SBATCH --time=8:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

pwd; hostname; date

whisper_path=/blue/liu.ying/conda/envs/whisper/bin
export PATH=$whisper_path:$PATH

cd /blue/liu.ying/transcription_bottleneck/

'''

## TODO: Modify cd above for your individual experiments

pretrained_model = "wav2vec2-large-xlsr-53" #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b
 

with open('pbs/' + lg + '_random.pbs', 'w') as f:
	first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + lg + '_' + size + '_random     # Job name'

	f.write(first_string + '\n')
	f.write(second_string + '\n')

	for seed in ['1', '2']:
		f.write('python scripts/train_random_w2v.py --lang ' + lg + ' --seed ' + seed + ' --pretrained_model ' + pretrained_model + '\n')
		f.write('\n')
		f.write('python scripts/eval_random_w2v.py --lang ' + lg + ' --seed ' + seed + ' --pretrained_model ' + pretrained_model + '\n')
		f.write('\n')

					
		f.write('\n')
