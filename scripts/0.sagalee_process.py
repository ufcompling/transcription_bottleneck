## Data source: https://www.openslr.org/157/

import io, os

os.system('mkdir ../asr_corpora/Sagalee/')
os.system('mkdir ../asr_corpora/Sagalee/wav/')
os.system('mkdir ../asr_corpora/Sagalee/txt/')

os.system('mv ../asr_resource/SLR157/Sagalee/train/*/*.wav ../asr_corpora/Sagalee/wav/')
os.system('mv ../asr_resource/SLR157/Sagalee/train/*/*.txt ../asr_corpora/Sagalee/txt/')
os.system('mv ../asr_resource/SLR157/Sagalee/dev/*/*.wav ../asr_corpora/Sagalee/wav/')
os.system('mv ../asr_resource/SLR157/Sagalee/dev/*/*.txt ../asr_corpora/Sagalee/txt/')
os.system('mv ../asr_resource/SLR157/Sagalee/test/*/*.wav ../asr_corpora/Sagalee/wav/')
os.system('mv ../asr_resource/SLR157/Sagalee/test/*/*.txt ../asr_corpora/Sagalee/txt/') 