## This script processes data from /blue/liu.ying/asr_resource/BembaSpeech to /blue/liu.ying/asr_corpora/Bemba

import io, os
import pandas as pd

os.system('mkdir ../asr_corpora/Bemba/')
os.system('mkdir ../asr_corpora/Bemba/wav/')
os.system('mkdir ../asr_corpora/Bemba/txt/')
os.system('mv ../asr_resource/BembaSpeech/bem/audio/*.wav ../asr_corpora/Bemba/wav/')

def read_file(file_path):
    data = {}
    original_data = pd.read_csv(file_path, sep = '\t')  
    audio_list = original_data['audio'].tolist()
    transcript_list = original_data['sentence'].tolist()
    for i in range(len(audio_list)):
        data[audio_list[i].split('.')[0]] = transcript_list[i]
    return data

train_data = read_file('../asr_resource/BembaSpeech/bem/train.tsv')
dev_data = read_file('../asr_resource/BembaSpeech/bem/dev.tsv')
test_data = read_file('../asr_resource/BembaSpeech/bem/test.tsv')

for k, v in train_data.items():
    with io.open('../asr_corpora/Bemba/txt/' + k + '.txt', 'w', encoding='utf-8') as f:
        f.write(v + '\n')

for k, v in dev_data.items():
    with io.open('../asr_corpora/Bemba/txt/' + k + '.txt', 'w', encoding='utf-8') as f:
        f.write(v + '\n')

for k, v in test_data.items():
    with io.open('../asr_corpora/Bemba/txt/' + k + '.txt', 'w', encoding='utf-8') as f:
        f.write(v + '\n')