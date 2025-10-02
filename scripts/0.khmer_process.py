## This script processes data from /blue/liu.ying/asr_resource/SLR42/ to /blue/liu.ying/asr_corpora/Khmer
## Data source: https://www.openslr.org/42/

import io, os

os.system('mkdir ../asr_corpora/Khmer/')
os.system('mkdir ../asr_corpora/Khmer/wav/')
os.system('mkdir ../asr_corpora/Khmer/txt/')
os.system('mv ../asr_resource/SLR42/km_kh_male/wavs/*.wav ../asr_corpora/Khmer/wav/')

with open('/blue/liu.ying/asr_resource/SLR42/km_kh_male/line_index.tsv') as f:
    for line in f:
        toks = line.strip().split()
        filename = toks[0] + '.txt'                  
        transc = toks[1 : :]
        with open('../asr_corpora/Khmer/txt/' + filename, 'w') as outfile:
            outfile.write(' '.join(w for w in transc) + '\n')


