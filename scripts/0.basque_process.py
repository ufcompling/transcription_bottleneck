## This script processes data from /blue/liu.ying/asr_resource/SLR76/ to /blue/liu.ying/asr_corpora/Basque
## Data source: https://www.openslr.org/76/

import io, os

os.system('mkdir ../asr_corpora/Basque/')
os.system('mkdir ../asr_corpora/Basque/wav/')
os.system('mkdir ../asr_corpora/Basque/txt/')
os.system('mv ../asr_resource/SLR76/*.wav ../asr_corpora/Basque/wav/')

with open('/blue/liu.ying/asr_resource/SLR76/line_index_male.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Basque/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')

with open('/blue/liu.ying/asr_resource/SLR76/line_index_female.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : ]
        with open('../asr_corpora/Basque/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')
