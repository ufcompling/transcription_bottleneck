## Data source: https://github.com/Bartelds/asr-augmentation?tab=readme-ov-file

# Gronings, Nasal, and Besemah

import io, os
import pandas as pd

os.system('mkdir ../asr_corpora/Gronings/')
os.system('mkdir ../asr_corpora/Gronings/wav/')
os.system('mkdir ../asr_corpora/Gronings/txt/')
os.system('mv ../asr_resource/20221103_gos/clips-clean/*.wav ../asr_corpora/Gronings/wav/')

os.system('mkdir ../asr_corpora/Nasal/')
os.system('mkdir ../asr_corpora/Nasal/wav/')
os.system('mkdir ../asr_corpora/Nasal/txt/')
os.system('mv ../asr_resource/20221014_nasal/data/*.wav ../asr_corpora/Nasal/wav/')

os.system('mkdir ../asr_corpora/Besemah/')
os.system('mkdir ../asr_corpora/Besemah/wav/')
os.system('mkdir ../asr_corpora/Besemah/txt/')
os.system('mv ../asr_resource/20221102_besemah/clips/*.wav ../asr_corpora/Besemah/wav/')

def output_transcripts(directory, lang):
    for file in os.listdir(directory):
        if file.endswith('.tsv'):
            data = pd.read_csv(os.path.join(directory, file), sep='\t')
            path_list = data['path'].tolist()
            text_list = data['text'].tolist()
            transcript_filename_list = []
            for i in range(len(path_list)):
                transcript_filename = path_list[i].split('/')[1].split('.')[0] + '.txt'
                transcript = text_list[i]
                with open('../asr_corpora/' + lang + '/txt/' + transcript_filename, 'w', encoding='utf-8') as f:
                    f.write(transcript + '\n')

output_transcripts('../asr_resource/20221103_gos/', 'Gronings')
output_transcripts('../asr_resource/20221014_nasal/', 'Nasal')
output_transcripts('../asr_resource/20221102_besemah/', 'Besemah')