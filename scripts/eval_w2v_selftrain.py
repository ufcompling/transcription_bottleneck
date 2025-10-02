import argparse
import os, re, sys
from transformers import Wav2Vec2Processor, AutoModelForCTC
import soundfile as sf
from datasets import load_metric
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
parser.add_argument("--pretrained_model", type=str, default="wav2vec2-large-xlsr-53") #wav2vec2-xls-r-300M, wav2vec2-xls-r-1b, wav2vec2-xls-r-2b

args = parser.parse_args()
lang = (args.lang).capitalize()
data_path = args.data_path + lang + '/'
data_path = data_path.replace(u'\xa0', u'')
size = args.size 
select_interval = args.interval
select = args.select
method = args.method
pretrained_model = args.pretrained_model

sub_datadir = data_path + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/'

lm_model= sub_datadir + pretrained_model + "_lm.arpa"
if select == 'all':
	lm_model = data_path + 'lm.arpa'

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�]'


lm = "no_lm"

# checkpoint = sorted([x for x in os.listdir("./xlsr53_{}/".format(lang+oov_rate)) if "checkpoint" in x], reverse=True)[0]
checkpoint = max([x for x in os.listdir('model/' + lang + '/' + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + pretrained_model + '/') if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_models = repo_name = 'model/' + lang + '/' + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + pretrained_model + '/'
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

to_predict = []

to_predict = ['test']

print('To predict: ', to_predict)

for target in to_predict:
	target_data = pd.read_csv(data_path + 'test.csv')

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

	for i in tqdm(range(0, len(signals), 1)):
		sig = signals[i:i+1]
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

		for ids in logits:
	#	beam_string = decoder.decode(ids).lower()
			beam_info = decoder.decode_beams(ids)[0]
			beam_string = beam_info[0]
			decoded.append(beam_string)

		preds = preds + decoded
		if method == 'al' and target == 'select':
			logit_confidence_scores += logit_scores
			lm_confidence_scores += lm_scores

	with open(sub_datadir + pretrained_model + '_' + target + '_preds.txt', 'w') as f:
		for pred in preds:
			f.write(pred + '\n')

	with open(sub_datadir + pretrained_model + '_' + target + '_eval.txt', 'w') as f:
		f.write('WER: ' + str(wer_metric.compute(predictions=preds, references=sentences)) + '\n')
		f.write('CER: ' + str(cer_metric.compute(predictions=preds, references=sentences)) + '\n')

