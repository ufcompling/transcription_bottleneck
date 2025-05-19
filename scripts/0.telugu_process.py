## This script processes data from /blue/liu.ying/asr_resource/SLR66/ to /blue/liu.ying/asr_corpora/Telugu
## Data source: https://www.openslr.org/66/

import io, os

os.system('mkdir ../asr_corpora/Telugu/')
os.system('mkdir ../asr_corpora/Telugu/wav/')
os.system('mkdir ../asr_corpora/Telugu/txt/')
os.system('mv ../asr_resource/SLR66/*.wav ../asr_corpora/Telugu/wav/')

with open('/blue/liu.ying/asr_resource/SLR66/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Telugu/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR66/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Telugu/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
