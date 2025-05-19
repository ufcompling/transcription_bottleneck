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
parser.add_argument("--method", type=str, default="al")
parser.add_argument("--seed", type=str, default="1")
parser.add_argument("--pretrained_model", type=str, default="wav2vec2-xls-r-2b") #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b wav2vec2-large-xlsr-53

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
	info = str(target_path_list[i]) + '\t' + str(target_transcript_list[i]) + '\t' + str(target_dur_list[i]) + '\t' + str(target_speaker_list[i])
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
		print(file, 'test')
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

		signal, sr = '_', '_'
		print(wav_path, 'test')
		try:
			signal, sr = sf.read(wav_path)
		except:
			print('Test', wav_path)

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
			print('Predictions: ', preds)
			for i in range(len(preds)):
				pred = preds[i].replace("\n", " ")
				ref = sentences[i].replace("\n", " ")
				outfile.write("prediction:  " + pred + "\n")
				outfile.write("reference:   " + ref + "\n")

# Generating prediction for individual select utterances
if select != 'all':
	for file in os.listdir(sub_datadir + 'select_individual/'):
		if file.endswith('.txt'):
			print(file, 'select')
			filename = file.split('.')[0]
			target_data_individual = ''
			with open(sub_datadir + 'select_individual/' + file) as f:
				for line in f:
					target_data_individual = line.strip().split('\t')

			wav_path = target_data_individual[0]
			if 'wav' not in wav_path and 'flac' not in wav_path:
				with open(sub_datadir + 'select_individual/' + file) as f:
					for line in f:
						wav_path = line.strip().split('\t')[0]

			wav_dur = target_data_individual[2]
			wav_speaker = target_data_individual[3]
			data = []
			transc = target_data_individual[1]
			transc = re.sub(chars_to_remove_regex, '', transc).lower()

			print(wav_path, 'select')
			signal, sr = '_', '_'
			try:
				signal, sr = sf.read(wav_path)
			except:
				print('Select', wav_path)
		
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
					
					beam_logit_score = beam_info[-2]
					beam_lm_score = beam_info[-1]
					
					logit_scores.append(beam_logit_score)
					lm_scores.append(beam_lm_score)
				
				preds = preds + decoded
				logit_confidence_scores += logit_scores
				lm_confidence_scores += lm_scores
			
			with open(sub_datadir + 'select_predictions/' + filename + '_confidence.txt', mode="w", encoding="utf-8") as outfile:
				for i in range(len(preds)):
					pred = preds[i].replace("\n", " ")
					ref = sentences[i].replace("\n", " ")
					logit_confidence_score = logit_confidence_scores[i]
					lm_confidence_score = lm_confidence_scores[i]
					outfile.write('path: ' + wav_path + '\n')
					outfile.write('speaker: ' + wav_speaker + '\n')
					outfile.write('duration: ' + wav_dur + '\n')
					outfile.write("prediction: " + pred + "\n")
					outfile.write("reference: " + ref + "\n")
					outfile.write("logit confidence score: " + str(float(logit_confidence_score)) + '\n')
					outfile.write("lm confidence score: " + str(float(lm_confidence_score)) + '\n')
				
## Computing WER for the test set
test_data = pd.read_csv(data_path + 'test_' + seed + '.csv')
test_sentences = []
test_predictions = []
for file in os.listdir(sub_datadir + 'test_predictions/'):
	if file.endswith('.txt'):
		with open(sub_datadir + 'test_predictions/' + file) as f:
			try:
				for line in f:
					if line.startswith('prediction:'):
						prediction = line.strip().split(': ')[1]
						test_predictions.append(prediction)
					if line.startswith('reference:'):
						reference = line.strip().split(': ')[1]
						test_sentences.append(reference)
			except:
				print(file)
				pass

print(len(test_sentences))
print(len(test_predictions))
with open(sub_datadir + pretrained_model + '_test_eval.txt', 'w') as f:
	f.write('WER: ' + str(wer_metric.compute(predictions=test_predictions, references=test_sentences)) + '\n')
	f.write('CER: ' + str(cer_metric.compute(predictions=test_predictions, references=test_sentences)) + '\n')

## Generate confidence scores for select file
confidence_dict = {}
for file in os.listdir(sub_datadir + 'select_predictions/'):
	if file.endswith('.txt'):
		wav_path = ''
		wav_speaker = ''
		wav_dur = ''
		prediction = ''
		reference = ''
		logit_confidence_score = ''
		with open(sub_datadir + '/select_predictions/' + file) as f:
			try:
				for line in f:
					if line.startswith('path:'):
						wav_path = line.strip().split(': ')[1]
					if line.startswith('speaker:'):
						wav_speaker = line.strip().split(': ')[1]
					if line.startswith('duration:'):
						wav_dur = line.strip().split(': ')[1]
					if line.startswith('prediction:'):
						prediction = line.strip().split(': ')[1]
					if line.startswith('reference:'):
						reference = line.strip().split(': ')[1]
						if reference.startswith('pengerami tu diatur kena'):
							print(sub_datadir + '/select_predictions/' + file, wav_path)
					if 'logit confidence score' in line:
						logit_confidence_score = line.strip().split(': ')[1]
			except:
				print(file)
				pass
		confidence_dict[wav_path + '\t' + reference + '\t' + str(wav_dur) + '\t' + wav_speaker] = logit_confidence_score #/ len(sentences[i].split()) controlling for sentence length

sorted_confidence_dict = sorted(confidence_dict.items(), key = lambda item: item[1])
increment_data = []
increment_dur = 0
residual_data = []
residual_dur = 0
temp_dur = 0

z = 0
while increment_dur <= int(select_interval) * 60 and z < len(sorted_confidence_dict):
	tok = sorted_confidence_dict[z]
	info = tok[0].split('\t')
	dur = info[-2]
	temp_dur += float(dur)
	if temp_dur <= int(select_interval) * 60:
		increment_data.append(info)		
		increment_dur += float(dur)
	else:
		residual_data.append(info)
		residual_dur += float(dur)
	z += 1

print('Increment data size: ' + str(increment_dur / 60))
print('Residual data size: ' + str(residual_dur / 60))

increment_wav_path_list = [tok[0] for tok in increment_data]
increment_transcript_list = [tok[1] for tok in increment_data]
increment_dur_list = [tok[-2] for tok in increment_data]
increment_speaker_list = [tok[-1] for tok in increment_data]

increment_output = pd.DataFrame({'path': increment_wav_path_list,
	'transcript': increment_transcript_list,
	'duration': increment_dur_list,
	'speaker': increment_speaker_list})

increment_output.to_csv(sub_datadir + pretrained_model + '_increment.input', index = False)

residual_wav_path_list = [tok[0] for tok in residual_data]
residual_transcript_list = [tok[1] for tok in residual_data]
residual_dur_list = [tok[-2] for tok in residual_data]
residual_speaker_list = [tok[-1] for tok in residual_data]

residual_output = pd.DataFrame({'path': residual_wav_path_list,
	'transcript': residual_transcript_list,
	'duration': residual_dur_list,
	'speaker': residual_speaker_list})

residual_output.to_csv(sub_datadir + pretrained_model + '_residual.input', index = False)

