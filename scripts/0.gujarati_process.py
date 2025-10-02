## This script processes data from /blue/liu.ying/asr_resource/SLR78/ to /blue/liu.ying/asr_corpora/Gujarati
## Data source: https://www.openslr.org/78/

import io, os

os.system('mkdir ../asr_corpora/Gujarati/')
os.system('mkdir ../asr_corpora/Gujarati/wav/')
os.system('mkdir ../asr_corpora/Gujarati/txt/')
os.system('mv ../asr_resource/SLR78/*.wav ../asr_corpora/Gujarati/wav/')

with open('/blue/liu.ying/asr_resource/SLR78/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Gujarati/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR78/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Gujarati/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
