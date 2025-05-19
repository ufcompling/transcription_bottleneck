# Data source: https://www.openslr.org/22/

import io, os

os.system('mkdir ../asr_corpora/Uyghur/')
os.system('mkdir ../asr_corpora/Uyghur/wav/')
os.system('mkdir ../asr_corpora/Uyghur/txt/')

os.system('mv ../asr_resource/SLR22/Uyghur/data_thuyg20/data/*.wav ../asr_corpora/Uyghur/wav/')

def collect_transcript(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            audio_filename = line[0].split('/')[-1]
            transcript_filename = audio_filename.split('.')[0] + '.txt'
            transcript = line[1:]
            with open('../asr_corpora/Uyghur/txt/' + transcript_filename, 'w', encoding='utf-8') as f:
                f.write(' '.join(w for w in transcript) + '\n')

file1 = '../asr_resource/SLR22/Uyghur/data_thuyg20/trans/trans.cv'
file2 = '../asr_resource/SLR22/Uyghur/data_thuyg20/trans/trans.train'
file3 = '../asr_resource/SLR22/Uyghur/data_thuyg20/trans/trans.test'

collect_transcript(file1)
collect_transcript(file2)
collect_transcript(file3)
