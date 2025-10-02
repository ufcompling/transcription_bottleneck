## This script processes data from /blue/liu.ying/asr_resource/SLR52/ to /blue/liu.ying/asr_corpora/Sinhala
## Data source: https://www.openslr.org/52/

import io, os

os.system('mkdir ../asr_corpora/Sinhala/')
os.system('mkdir ../asr_corpora/Sinhala/wav/')
os.system('mkdir ../asr_corpora/Sinhala/txt/')

for directory in os.listdir('../asr_resource/SLR52/asr_sinhala/data/'):
    os.system('mv ../asr_resource/SLR52/asr_sinhala/data/' + directory + '/*.flac ../asr_corpora/Sinhala/wav/')

with open('/blue/liu.ying/asr_resource/SLR52/asr_sinhala/utt_spk_text.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[2 : ]
        with open('../asr_corpora/Sinhala/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
