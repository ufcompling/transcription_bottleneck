## This script processes data from /blue/liu.ying/asr_resource/SLR63/ to /blue/liu.ying/asr_corpora/Malayalam
## Data source: https://www.openslr.org/63/

import io, os

os.system('mkdir ../asr_corpora/Malayalam/')
os.system('mkdir ../asr_corpora/Malayalam/wav/')
os.system('mkdir ../asr_corpora/Malayalam/txt/')
os.system('mv ../asr_resource/SLR63/*.wav ../asr_corpora/Malayalam/wav/')

with open('/blue/liu.ying/asr_resource/SLR63/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Malayalam/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR63/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Malayalam/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
