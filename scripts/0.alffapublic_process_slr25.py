## This script processes data from /blue/liu.ying/asr_resource/ALFFA_PUBLIC to /blue/liu.ying/asr_corpora/Iban
## Languages of interest: Amharic, Fongbe, Swahili, Wolof
## Data source: https://www.openslr.org/25/

import io, os

languages_of_interest = ['Amharic', 'Fongbe', 'Swahili', 'Wolof']
for lang in languages_of_interest:
    os.system('mkdir ../asr_corpora/' + lang + '/')
    os.system('mkdir ../asr_corpora/' + lang + '/wav/')
    os.system('mkdir ../asr_corpora/' + lang + '/txt/')

## Amharic
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/AMHARIC/data/train/wav/*.wav ../asr_corpora/Amharic/wav/')
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/AMHARIC/data/test/wav/*.wav ../asr_corpora/Amharic/wav/')

def collect_transcript(file_path, lang):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split()
            filename = line[0] + '.txt'
            transcript = line[1 : ]
            with io.open('../asr_corpora/' + lang + '/txt/' + filename, 'w', encoding='utf-8') as f:
                f.write(' '.join(w for w in transcript) + '\n')

train_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/AMHARIC/data/train/text'
test_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/AMHARIC/data/test/text'

collect_transcript(train_text, 'Amharic')
collect_transcript(test_text, 'Amharic')

## Fongbe (too many short utterances)
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/FONGBE/data/train/wav/*/*.wav ../asr_corpora/Fongbe/wav/')
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/FONGBE/data/test/wav/*/*.wav ../asr_corpora/Fongbe/wav/')

train_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/FONGBE/data/train/text'
test_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/FONGBE/data/test/text'

collect_transcript(train_text)
collect_transcript(test_text)

## Swahili
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/SWAHILI/data/train/wav/*/*.wav ../asr_corpora/Swahili/wav/')
os.system('mv ../asr_resource/SLR25/ALFFA_PUBLIC/ASR/SWAHILI/data/test/wav5/*/*.wav ../asr_corpora/Swahili/wav/')

train_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/SWAHILI/data/train/text'
test_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/SWAHILI/data/test/text'

collect_transcript(train_text, 'Swahili')
collect_transcript(test_text, 'Swahili')

## Wolof

train_directory = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/WOLOF/data/train/'
dev_directory = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/WOLOF/data/dev/'
test_directory = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/WOLOF/data/test/'
for d in os.listdir(train_directory):
    try:
        os.system('mv ' + train_directory + d + '/*.wav ../asr_corpora/Wolof/wav/')
    except:
        print(d)

for d in os.listdir(dev_directory):
    try:
        os.system('mv ' + dev_directory + d + '/*.wav ../asr_corpora/Wolof/wav/')
    except:
        pass

for d in os.listdir(test_directory):
    try:
        os.system('mv ' + test_directory + d + '/*.wav ../asr_corpora/Wolof/wav/')
    except:
        pass

train_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/WOLOF/data/train/text'
#dev_text = '../asr_resource/ALFFA_PUBLIC/ASR/WOLOF/data/dev/text'  # no wav files
test_text = '../asr_resource/SLR25/ALFFA_PUBLIC/ASR/WOLOF/data/test/text'

collect_transcript(train_text, 'Wolof')
collect_transcript(test_text, 'Wolof')