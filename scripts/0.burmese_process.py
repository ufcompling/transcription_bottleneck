## This script processes data from /blue/liu.ying/asr_resource/SLR80/ to /blue/liu.ying/asr_corpora/Burmese
## Data source: https://www.openslr.org/80/

import io, os

os.system('mkdir ../asr_corpora/Burmese/')
os.system('mkdir ../asr_corpora/Burmese/wav/')
os.system('mkdir ../asr_corpora/Burmese/txt/')
os.system('mv ../asr_resource/SLR80/*.wav ../asr_corpora/Burmese/wav/')

with open('/blue/liu.ying/asr_resource/SLR80/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Burmese/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
