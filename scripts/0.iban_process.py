## This script processes data from /blue/liu.ying/asr_resource/iban to /blue/liu.ying/asr_corpora/Iban

import io, os

os.system('mkdir ../asr_corpora/Iban/')
os.system('mkdir ../asr_corpora/Iban/wav/')
os.system('mkdir ../asr_corpora/Iban/txt/')
os.system('mv ../asr_resource/iban/data/wav/*/*.wav ../asr_corpora/Iban/wav/')

with open('../asr_resource/iban/data/train/train_text', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        filename = line[0] + '.txt'
        transcript = line[1]
        with io.open('../asr_corpora/Iban/txt/' + filename, 'w', encoding='utf-8') as f:
            f.write(transcript + '\n')

with open('../asr_resource/iban/data/test/test_text', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        filename = line[0] + '.txt'
        transcript = line[1]
        with io.open('../asr_corpora/Iban/txt/' + filename, 'w', encoding='utf-8') as f:
            f.write(transcript + '\n')
