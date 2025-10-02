import argparse
import os, re, sys
from transformers import Wav2Vec2Processor, AutoModelForCTC
import soundfile as sf
#from datasets import load_metric
from jiwer import wer
import torch
import torchaudio
import torchaudio.sox_effects as ta_sox
from pyctcdecode import build_ctcdecoder
import numpy as np
from tqdm import tqdm
import shutil
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--lang", type=str, default="hupa")
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--size", type=str, default="30") ## initial training size
parser.add_argument("--interval", type=str, default="5")
parser.add_argument("--select", type=str, default="0")
parser.add_argument("--method", type=str, default="random")
parser.add_argument("--seed", type=str, default="1")
parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-xlsr-53") #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b

args = parser.parse_args()
lang = args.lang
if lang != 'isiXhosa':
	lang = (args.lang).capitalize()
data_path = args.data_path + lang + '/'
data_path = data_path.replace(u'\xa0', u'')

size = args.size 
select_interval = args.interval
select = args.select
method = args.method
seed = args.seed
pretrained_model = args.pretrained_model

sub_datadir = data_path + '/' + size + '/' + method + '/' + seed + '/' + select_interval + '/select' + select + '/'

# Splitting test csv file into individual txt files
if os.path.exists(data_path + 'test_individual/') == True:
	os.system('rm -r ' + data_path + 'test_individual/')

os.system('mkdir ' + data_path + 'test_individual/')

target_data = pd.read_csv(data_path + 'test_' + seed + '.csv')
target_path_list = target_data['path'].tolist()
target_transcript_list = target_data['transcript'].tolist()
target_dur_list = target_data['duration'].tolist()
target_speaker_list = target_data['speaker'].tolist()
for i in range(len(target_path_list)):
	info = target_path_list[i] + '\t' + target_transcript_list[i] + '\t' + str(target_dur_list[i]) + '\t' + target_speaker_list[i]
	with open(data_path + 'test_individual/' + str(i) + '.txt', 'w') as f:
		f.write(info + '\n')

# Splitting select csv file into individual txt files
if os.path.exists(sub_datadir + 'select_individual/') == True:
	os.system('rm -r ' + sub_datadir + 'select_individual/')

os.system('mkdir ' + sub_datadir + 'select_individual/')

target_data = pd.read_csv(sub_datadir + pretrained_model + '_select.' + size + '.input')
target_path_list = target_data['path'].tolist()
target_transcript_list = target_data['transcript'].tolist()
target_dur_list = target_data['duration'].tolist()
target_speaker_list = target_data['speaker'].tolist()
for i in range(len(target_path_list)):
	info = target_path_list[i] + '\t' + target_transcript_list[i] + '\t' + str(target_dur_list[i]) + '\t' + target_speaker_list[i]
	with open(sub_datadir + 'select_individual/' + str(i) + '.txt', 'w') as f:
		f.write(info + '\n')

# Create a folder to put individual predictions
if os.path.exists(sub_datadir + 'test_predictions/') == True:
	os.system('rm -r ' + sub_datadir + 'test_predictions/')
os.system('mkdir ' + sub_datadir + 'test_predictions/')

if os.path.exists(sub_datadir + 'select_predictions/') == True:
	os.system('rm -r' + sub_datadir + 'select_predictions/')
os.system('mkdir ' + sub_datadir + 'select_predictions/')
		
lm_model= sub_datadir + pretrained_model + "_lm.arpa"
if select == 'all':
	lm_model = data_path + 'lm.arpa'

import evaluate
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�]'

# checkpoint = sorted([x for x in os.listdir("./xlsr53_{}/".format(lang+oov_rate)) if "checkpoint" in x], reverse=True)[0]
checkpoint = max([x for x in os.listdir('model/' + lang + '/' + size + '/' + method + '/' + seed + '/' + select_interval + '/select' + select + '/' + pretrained_model + '/') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_models = repo_name = 'model/' + lang + '/' + size + '/' + method + '/' + seed + '/' + select_interval + '/select' + select + '/' + pretrained_model + '/'
path_checkpoint = path_models + checkpoint
print('Checkpoint: ' + path_checkpoint)
print('\n')

if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
	shutil.copy(path_models+"tokenizer_config.json", path_checkpoint+"/tokenizer_config.json")
	shutil.copy(path_models+"vocab.json", path_checkpoint+"/vocab.json")

model = AutoModelForCTC.from_pretrained(path_checkpoint).to("cuda")#change

processor = Wav2Vec2Processor.from_pretrained(path_checkpoint)
vocab = processor.tokenizer.get_vocab()

vocab[' '] = vocab['|']
del vocab[' ']
sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

decoder = build_ctcdecoder(
	list(sorted_dict.keys()),
	lm_model,
	alpha = 0.5,
	beta = 1.5
)

# Generating prediction for individual test utterances
for file in os.listdir(data_path + 'test_individual/'):
	if file.endswith('.txt'):
		filename = file.split('.')[0]
		target_data_individual = ''
		with open(data_path + 'test_individual/' + file) as f:
			for line in f:
				target_data_individual = line.strip().split('\t')
		
		wav_path = target_data_individual[0]
		wav_dur = target_data_individual[2]
		wav_speaker = target_data_individual[3]
		data = []
		transc = target_data_individual[1]
		transc = re.sub(chars_to_remove_regex, '', transc).lower()

		signal, sr = sf.read(wav_path)
		
		entry = {}
				
		entry["sentence"] = transc			
		entry["audio"] = {"sampling_rate" : sr, "array" : signal}
		entry['path'] = wav_path
		data.append(entry)
		
		signals = [x["audio"]["array"] for x in data]
		sentences = [x["sentence"] for x in data]

		preds = []
		logit_confidence_scores = []
		lm_confidence_scores = []
		
		for i in tqdm(range(0, len(signals), 1)):
			sig = signals[i:i+1]
			inputs = processor(sig, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
			if inputs.shape[-1] == 2:
				sig = [np.average(sig[0], axis = 1)]
				inputs = processor(sig, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
			else:
				pass
			
			with torch.no_grad():
				logits = model(inputs).logits.to("cpu").numpy()
			
			decoded = []
			logit_scores = []
			lm_scores = []
	
			for ids in logits:
				beam_info = decoder.decode_beams(ids)[0]
				beam_string = beam_info[0]
				decoded.append(beam_string)
	
			preds = preds + decoded

		with open(sub_datadir + 'test_predictions/' + filename + '.txt', 'w') as outfile:
			for i in range(len(preds)):
				pred = preds[i].replace("\n", " ")
				ref = sentences[i].replace("\n", " ")
				outfile.write("prediction:  " + pred + "\n")
				outfile.write("reference:   " + ref + "\n")

## Computing WER for the test set
test_data = pd.read_csv(data_path + 'test_' + seed + '.csv')
test_sentences = []
test_predictions = []
for file in os.listdir(sub_datadir + 'test_predictions/'):
	if file.endswith('.txt'):
		with open(sub_datadir + 'test_predictions/' + file) as f:
			for line in f:
				if 'prediction' in line:
					prediction = line.strip().split(': ')[1]
					test_predictions.append(prediction)
				if 'reference' in line:
					reference = line.strip().split(': ')[1]
					test_sentences.append(reference)

print(len(test_sentences))
print(len(test_predictions))
with open(sub_datadir + pretrained_model + '_test_eval.txt', 'w') as f:
	f.write('WER: ' + str(wer_metric.compute(predictions=test_predictions, references=test_sentences)) + '\n')
	f.write('CER: ' + str(cer_metric.compute(predictions=test_predictions, references=test_sentences)) + '\n')
<<<<<<< HEAD
=======

>>>>>>> 644c7d9e26d72a5fe24f6887da0745ffc880f538
