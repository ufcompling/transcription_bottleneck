## This script processes data from /blue/liu.ying/asr_resource/SLR53/ to /blue/liu.ying/asr_corpora/Bengali
## Data source: https://www.openslr.org/53/

import io, os

os.system('mkdir ../asr_corpora/Bengali/')
os.system('mkdir ../asr_corpora/Bengali/wav/')
os.system('mkdir ../asr_corpora/Bengali/txt/')

for directory in os.listdir('../asr_resource/SLR53/asr_bengali/data/'):
    os.system('mv ../asr_resource/SLR53/asr_bengali/data/' + directory + '/*.flac ../asr_corpora/Bengali/wav/')

with open('/blue/liu.ying/asr_resource/SLR53/asr_bengali/utt_spk_text.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[2 : ]
        with open('../asr_corpora/Bengali/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
