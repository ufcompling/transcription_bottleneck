## This script processes data from /blue/liu.ying/asr_resource/SLR54/ to /blue/liu.ying/asr_corpora/Nepali
## Data source: https://www.openslr.org/54/

import io, os

os.system('mkdir ../asr_corpora/Nepali/')
os.system('mkdir ../asr_corpora/Nepali/wav/')
os.system('mkdir ../asr_corpora/Nepali/txt/')

for directory in os.listdir('../asr_resource/SLR54/asr_nepali/data/'):
    os.system('mv ../asr_resource/SLR54/asr_nepali/data/' + directory + '/*.flac ../asr_corpora/Nepali/wav/')

with open('/blue/liu.ying/asr_resource/SLR54/asr_nepali/utt_spk_text.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[2 : ]
        with open('../asr_corpora/Nepali/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
