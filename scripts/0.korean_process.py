## This script processes data from /blue/liu.ying/asr_resource/SLR40/ to /blue/liu.ying/asr_corpora/Korean
## Data source: https://www.openslr.org/40/

import io, os

os.system('mkdir ../asr_corpora/Korean/')
os.system('mkdir ../asr_corpora/Korean/wav/')
os.system('mkdir ../asr_corpora/Korean/txt/')
os.system('mv ../asr_resource/SLR40/train_data_01/003/*/*.flac ../asr_corpora/Korean/wav/')
os.system('mv ../asr_resource/SLR40/test_data_01/003/*/*.flac ../asr_corpora/Korean/wav/')


for split in ['train_data_01', 'test_data_01']:
    for directory in os.listdir('../asr_resource/SLR40/' + split + '/003/'):
        for file in os.listdir('../asr_resource/SLR40/' + split + '/003/' + directory):
            if file.endswith('txt'):
                with open('../asr_resource/SLR40/' + split + '/003/' + directory + '/' + file) as f:
                    for line in f:
                        toks = line.strip().split()
                        filename = toks[0] + '.txt'                  
                        transc = toks[1 : :]
                        with open('../asr_corpora/Korean/txt/' + filename, 'w') as outfile:
                            outfile.write(' '.join(w for w in transc) + '\n')