## Data source: https://www.openslr.org/32/
## This script processes data from /blue/liu.ying/asr_resource/ to /blue/liu.ying/asr_corpora/

import io, os

languages_of_interest = ['Afrikaans', 'Sesotho', 'Setswana', 'isiXhosa']
for lang in languages_of_interest:
    os.system('mkdir ../asr_corpora/' + lang + '/')
    os.system('mkdir ../asr_corpora/' + lang + '/wav/')
    os.system('mkdir ../asr_corpora/' + lang + '/txt/')

def process_data(directory, lang):
    transcript_path = ''
    wav_path = ''
    for d in os.listdir(directory + 'za/'):
        transcript_path = directory + 'za/' + d + '/line_index.tsv'       
        wav_path = directory + 'za/' + d + '/wavs/'
    os.system('mv ' + wav_path + '*.wav ../asr_corpora/' + lang + '/wav/')
    with open(transcript_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.split('\t')
            filename = line[0] + '.txt'
            transcript = line[1:]
            with io.open('../asr_corpora/' + lang + '/txt/' + filename, 'w', encoding='utf-8') as f:
                f.write(' '.join(w for w in transcript) + '\n')

process_data('../asr_resource/SLR32/af_za/', 'Afrikaans')
process_data('../asr_resource/SLR32/st_za/', 'Sesotho')
process_data('../asr_resource/SLR32/tn_za/', 'Setswana')
process_data('../asr_resource/SLR32/xh_za/', 'isiXhosa')