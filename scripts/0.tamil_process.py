## This script processes data from /blue/liu.ying/asr_resource/SLR65/ to /blue/liu.ying/asr_corpora/Tamil
## Data source: https://www.openslr.org/65/

import io, os

os.system('mkdir ../asr_corpora/Tamil/')
os.system('mkdir ../asr_corpora/Tamil/wav/')
os.system('mkdir ../asr_corpora/Tamil/txt/')
os.system('mv ../asr_resource/SLR65/*.wav ../asr_corpora/Tamil/wav/')

with open('/blue/liu.ying/asr_resource/SLR65/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Tamil/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR65/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Tamil/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
