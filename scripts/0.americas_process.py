## This script processes data from /blue/liu.ying/asr_resource/Quechua to /blue/liu.ying/asr_corpora/Quechua
## Data from AmericasNLP 2022 shared task on ASR and machine translation: https://github.com/AmericasNLP/americasnlp2022?tab=readme-ov-file
## Languages of interest: Quechua, Bribri, Guarani, Kotiria, Waikhana
## Usage: e.g., python3 scripts/0.americas_process.py Quechua

import io, os, sys
import pandas as pd

lang = sys.argv[1]
lang = lang.capitalize()

os.system('mkdir ../asr_corpora/' + lang + '/')
os.system('mkdir ../asr_corpora/' + lang + '/wav/')
os.system('mkdir ../asr_corpora/' + lang + '/txt/')
os.system('mv ../asr_resource/' + lang + '/train/*.wav ../asr_corpora/' + lang + '/wav/')
os.system('mv ../asr_resource/' + lang + '/dev/*.wav ../asr_corpora/' + lang + '/wav/')

def read_file(file_path):
    data = {}
    audio_list = []
    transcript_list = []
    try:
        original_data = pd.read_csv(file_path, sep = '\t')  
        audio_list = original_data['wav'].tolist()
        transcript_list = original_data['source_processed'].tolist()
    except:
        with open(file_path, 'r', encoding = 'utf-8') as f:
            for line in f:
                audio_list.append(line.split('\t')[0])
                transcript_list.append(line.split('\t')[1])
    for i in range(len(audio_list)):
        data[audio_list[i].split('.')[0]] = transcript_list[i]
    return data

train_data = read_file('../asr_resource/' + lang + '/train/meta.tsv')
dev_data = read_file('../asr_resource/' + lang + '/dev/meta.tsv')

for k, v in train_data.items():
    with io.open('../asr_corpora/' + lang + '/txt/' + k + '.txt', 'w', encoding='utf-8') as f:
        f.write(v + '\n')

for k, v in dev_data.items():
    with io.open('../asr_corpora/' + lang + '/txt/' + k + '.txt', 'w', encoding='utf-8') as f:
        f.write(v + '\n')

