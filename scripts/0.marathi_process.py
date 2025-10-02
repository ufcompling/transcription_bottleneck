## This script processes data from /blue/liu.ying/asr_resource/SLR64/ to /blue/liu.ying/asr_corpora/Malayalam
## Data source: https://www.openslr.org/64/

import io, os

os.system('mkdir ../asr_corpora/Marathi/')
os.system('mkdir ../asr_corpora/Marathi/wav/')
os.system('mkdir ../asr_corpora/Marathi/txt/')
os.system('mv ../asr_resource/SLR64/*.wav ../asr_corpora/Marathi/wav/')

with open('/blue/liu.ying/asr_resource/SLR64/line_index.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Marathi/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
