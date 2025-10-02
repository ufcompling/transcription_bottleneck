## This script processes data from /blue/liu.ying/asr_resource/SLR86/ to /blue/liu.ying/asr_corpora/Yoruba
## Data source: https://www.openslr.org/86/

import io, os

os.system('mkdir ../asr_corpora/Yoruba/')
os.system('mkdir ../asr_corpora/Yoruba/wav/')
os.system('mkdir ../asr_corpora/Yoruba/txt/')
os.system('mv ../asr_resource/SLR86/*.wav ../asr_corpora/Yoruba/wav/')

with open('/blue/liu.ying/asr_resource/SLR86/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Yoruba/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR86/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Yoruba/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
