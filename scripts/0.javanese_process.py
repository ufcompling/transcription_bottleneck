## This script processes data from /blue/liu.ying/asr_resource/SLR35/ to /blue/liu.ying/asr_corpora/Javanese
## Data source: https://www.openslr.org/35/

import io, os

os.system('mkdir ../asr_corpora/Javanese/')
os.system('mkdir ../asr_corpora/Javanese/wav/')
os.system('mkdir ../asr_corpora/Javanese/txt/')

for directory in os.listdir('../asr_resource/SLR35/asr_javanese/data/'):
    os.system('mv ../asr_resource/SLR35/asr_javanese/data/' + directory + '/*.flac ../asr_corpora/Javanese/wav/')

with open('/blue/liu.ying/asr_resource/SLR35/asr_javanese/utt_spk_text.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[2 : ]
        with open('/blue/liu.ying/asr_corpora/Javanese/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
