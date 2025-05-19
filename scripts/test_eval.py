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


to_predict = ['select']
if select == 'all':
	to_predict = ['test']

print('To predict: ', to_predict)

for target in to_predict:
	if target == 'test':
		target_data = pd.read_csv(data_path + 'test_' + seed + '.csv')
	else:
		target_data = pd.read_csv(sub_datadir + pretrained_model + '_select.' + size + '.input') ## to generate confidence scores for

	data = []

	GS_idx = 0

	target_path_list = target_data['path'].tolist()
	target_transcript_list = target_data['transcript'].tolist()
	target_dur_list = target_data['duration'].tolist()
	target_speaker_list = target_data['speaker'].tolist()

	print(target, 'duration: ', sum([float(dur) for dur in target_dur_list]) / 60)
	print('\n')

	for i in range(len(target_data)):
		transc = target_transcript_list[i]
		transc = re.sub(chars_to_remove_regex, '', transc).lower()

		wav_path = target_path_list[i]
	#	sr = 16000
		signal, sr = sf.read(wav_path)

	#	signal, sr = torchaudio.load(wav_path)

		entry = {}
				
		entry["sentence"] = transc			
		entry["audio"] = {"sampling_rate" : sr, "array" : signal}
		entry['path'] = wav_path
		data.append(entry)

	print("generate predictions")

	signals = [x["audio"]["array"] for x in data]
	sentences = [x["sentence"] for x in data]

	preds = []
	logit_confidence_scores = []
	lm_confidence_scores = []

	for i in tqdm(range(0, len(signals), 1)):
		sig = signals[i:i+1]
		print(i, sentences[i])
		inputs = processor(sig, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
		if inputs.shape[-1] == 2:
	#		print('Abnormal')
	#		print(sig)
			sig = [np.average(sig[0], axis = 1)]
	#		print(sig)
			inputs = processor(sig, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
		else:
			pass
	#		print(sig)
	#	print('A', inputs.shape)
		with torch.no_grad():
	#		print('B', model(inputs).logits.shape)
			logits = model(inputs).logits.to("cpu").numpy()
	#		print('C', logits.shape)
	#	print('\n')
		decoded = []
		logit_scores = []
		lm_scores = []
	
		for ids in logits:
	#	beam_string = decoder.decode(ids).lower()
			beam_info = decoder.decode_beams(ids)[0]
			beam_string = beam_info[0]
			decoded.append(beam_string)

			if target == 'select':
				beam_logit_score = beam_info[-2]
				beam_lm_score = beam_info[-1]

				logit_scores.append(beam_logit_score)
				lm_scores.append(beam_lm_score)

	#	beam_score = decoder.decode(ids, output_word_offsets=True).lm_score
	#	beam_score = beam_score / len(beam_string.split(" "))

	#	output = processor.batch_decode(ids, output_word_offsets=True)
	
		preds = preds + decoded
		if target == 'select':
			logit_confidence_scores += logit_scores
			lm_confidence_scores += lm_scores

	
#	confidence_scores += [score / len(t.split(" ")) for score, t in zip(output.lm_score, output.text)]

	with open(sub_datadir + pretrained_model + '_' + target + '_preds.txt', 'w') as f:
		for pred in preds:
			f.write(pred + '\n')

	if target == 'test':
		with open(sub_datadir + pretrained_model + '_' + target + '_eval.txt', 'w') as f:
			f.write('WER: ' + str(wer_metric.compute(predictions=preds, references=sentences)) + '\n')
			f.write('CER: ' + str(cer_metric.compute(predictions=preds, references=sentences)) + '\n')

	if target == 'select':
		confidence_dict = {}
		for i in range(len(sentences)):
			confidence_dict[target_path_list[i] + '\t' + sentences[i] + '\t' + str(target_dur_list[i]) + '\t' + target_speaker_list[i]] = logit_confidence_scores[i] #/ len(sentences[i].split()) controlling for sentence length

		sorted_confidence_dict = sorted(confidence_dict.items(), key = lambda item: item[1])
		increment_data = []
		increment_dur = 0
		residual_data = []
		residual_dur = 0
		temp_dur = 0

		z = 0
		while increment_dur <= int(select_interval) * 60 and z < len(sorted_confidence_dict):
			tok = sorted_confidence_dict[z]
			print(tok, z, len(sorted_confidence_dict))
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

	#	for tok in sorted_confidence_dict:
	#		info = tok[0].split('\t')
	#		dur = info[-2]
	#		if increment_dur <= int(select_interval) * 60:	
	#			increment_data.append(info)		
	#			increment_dur += float(dur)
	#		else:
	#			residual_data.append(info)
	#			residual_dur += float(dur)

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

		##### Output confidence scores as an individual file #####
		with open(sub_datadir + pretrained_model + "_confidence.txt", mode="w", encoding="utf-8") as tfile:
			for i in range(len(preds)):
				pred = preds[i].replace("\n", " ")
				ref = sentences[i].replace("\n", " ")
				logit_confidence_score = logit_confidence_scores[i]
				lm_confidence_score = lm_confidence_scores[i]
				tfile.write("prediction:  "+ pred+",")
				tfile.write("reference:   "+ ref+"\n")
				tfile.write("logit confidence score: " + str(float(logit_confidence_score)) + '\n')
				tfile.write("lm confidence score: " + str(float(lm_confidence_score)) + '\n')

'''
increment_output = pd.DataFrame({'path': path_list,
			'transcript': transcript_list,
			'duration': dur_list,
			'speaker': speaker_list})

increment_output.to_csv('temp.input', index = False)
'''
