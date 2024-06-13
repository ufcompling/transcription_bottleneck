import numpy as np
import re, os
import json
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from datasets import load_metric
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import soundfile as sf
import argparse
from tqdm import tqdm
from random import shuffle
import pandas as pd

if not os.path.exists('model/'):
	os.system('mkdir model/')

def read_audio(fname):
	""" Load an audio file and return PCM along with the sample rate """

	wav, sr = sf.read(fname)
	return wav, sr


chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\]\[]'

def clean_sent(sent):
	sent = re.sub(chars_to_remove_regex, '', sent).lower()
	return sent


@dataclass
class DataCollatorCTCWithPadding:
	"""
	Data collator that will dynamically pad the inputs received.
	Args:
		processor (:class:`~transformers.Wav2Vec2Processor`)
			The processor used for proccessing the data.
		padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
			Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
			among:
			* :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
			  sequence if provided).
			* :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
			  maximum acceptable input length for the model if that argument is not provided.
			* :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
			  different lengths).
	"""

	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]
	#	input_features = []
	#	label_features = []
	#	for feature in features:
	#		try:
	#			input_features.append({"input_features": feature["input_features"]})
	#			label_features.append({"input_ids": feature["labels"]})
	#		except:
	#			pass

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			return_tensors="pt",
		)
		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				return_tensors="pt",
			)

		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels

		return batch

def get_data_reg(data_path, file):

	## Define n-gram LM file
	lm_text = data_path + 'lm.txt'
	lm = data_path + 'lm.arpa'
	lm_vocab = data_path + 'vocab.txt'

	data_path = data_path + file
	data_path = data_path.replace(u'\xa0', u'')
	original_data = pd.read_csv(data_path)
	path_list = original_data['path'].tolist()
	transcript_list = original_data['transcript'].tolist()

	## Getting text data for training n-gram LM
	with open(lm_text, 'w') as f:
		for tok in transcript_list:
			f.write(tok + '\n')

	## Training an n-gram LM
	os.system('module load gcc')
	os.system('module load perl/5.20.0')
	os.system('/blue/liu.ying/asr_resource/kaldi/tools/srilm/bin/i686-m64/ngram-count -order 3 -unk -write-vocab ' + lm_vocab + ' -wbdiscount -text ' + lm_text + ' -lm ' + lm)

	## Getting audio data
	data = []
	long_wav = []
	long_transc = ""
	duration = 0
	words = []

	for i in range(len(path_list)):
		wav_path = path_list[i]
		transcript = clean_sent(transcript_list[i])
	#	sr = 16000
		wavform, sr = sf.read(wav_path)

		long_transc = long_transc + " " + transcript
				
		try:
			long_wav = np.concatenate([long_wav, wavform])
		except:
			pass

		if (len(long_wav) / sr)>=5:
			entry = {}
			duration += len(long_wav)
			words = words + long_transc.split()
			entry["sentence"] = long_transc.replace("\n", " ")				
			entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
			data.append(entry)

			long_wav = []
			long_transc = ""

	print('Duration:', duration / 16000)
	return data


def train(lang, data_path, size, select_interval, select, method, train_data, test_data, pretrained_model):

	# train_data = list(map(clean_sent, train_data))

	# test_data = list(map(clean_sent, test_data))
	print("creating vocab")
	vocab_train = set(y for x in train_data for y in x["sentence"])
	
	vocab_test = set(y for x in test_data for y in x["sentence"])
	vocab = vocab_train.union(vocab_test)
	if "\n" in vocab:
		vocab.remove("\n")
	vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	with open('vocab.json', 'w') as vocab_file:
		json.dump(vocab_dict, vocab_file, ensure_ascii=False)
	
	###Creation of the tokeniser###
	print("setting up tokeniser")
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	print("tokeniser saved")
	repo_name = 'model/' + lang + '/' + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + pretrained_model
	print(repo_name) 
	tokenizer.save_pretrained(repo_name)

	###Extraction of speech features###
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

	def compute_metrics(pred):
		pred_logits = pred.predictions
		pred_ids = np.argmax(pred_logits, axis=-1)

		pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

		pred_str = processor.batch_decode(pred_ids)

		label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

		wer = wer_metric.compute(predictions=pred_str, references=label_str)

		return {"wer": wer}

	def prepare_dataset(batch):

		try:
			audio = batch["audio"]


			batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0] #['input_features'][0] 
			batch["input_length"] = len(batch["input_values"])
		
			with processor.as_target_processor():
				batch["labels"] = processor(batch["sentence"]).input_ids
			return batch
		except:
			pass


	#### Setting up data for training###

	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
	wer_metric = load_metric("wer")
	train_data_temp = list(map(prepare_dataset, train_data))
	train_data = [tok for tok in train_data_temp if tok is not None]
	test_data_temp = list(map(prepare_dataset, test_data))
	test_data = [tok for tok in test_data_temp if tok is not None]
 
	print("preparing model")
	#### Training ####
	model = Wav2Vec2ForCTC.from_pretrained(
		"facebook/" + pretrained_model, 
		attention_dropout=0.0,
		hidden_dropout=0.0,
		feat_proj_dropout=0.0,
		mask_time_prob=0.05,
		layerdrop=0.0,
		ctc_loss_reduction="mean", 
		pad_token_id=processor.tokenizer.pad_token_id,
		vocab_size=len(processor.tokenizer),
	)
	model.config.ctc_zero_infinity = True #to prevent the loss of getting lost
	model.freeze_feature_extractor()
	from transformers import TrainingArguments

	epochs = 30
	batch_size = 16


	training_args = TrainingArguments(
	output_dir=repo_name,
	group_by_length=True,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=8,
	gradient_accumulation_steps=2,
	evaluation_strategy="steps",
	num_train_epochs=epochs,
	gradient_checkpointing=True,
	fp16=True,
	save_steps = 100,
	eval_steps = 100,
	logging_steps = 100,
#	warmup_steps = 500,
	learning_rate=3e-4,
	metric_for_best_model="wer",
	save_total_limit=2,
	greater_is_better=False,
	push_to_hub=False,
	)
	from transformers import Trainer
	print("start training")

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_data,
		eval_dataset=test_data,
		tokenizer=processor.feature_extractor
	)

	trainer.train()

def main():
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
	size = args.size 
	select_interval = args.interval
	select = args.select
	method = args.method
	pretrained_model = args.pretrained_model

	if not os.path.exists(data_path + size):
		os.system('mkdir ' + data_path + size)

	if not os.path.exists(data_path + size + '/' + method):
		os.system('mkdir ' + data_path + size + '/' + method)

	if not os.path.exists(data_path + size + '/' + method + '/' + select_interval):
		os.system('mkdir ' + data_path + size + '/' + method + '/' + select_interval)

	if not os.path.exists(data_path + size + '/' + method + '/' + select_interval + '/select' + select):
		os.system('mkdir ' + data_path + size + '/' + method + '/' + select_interval + '/select' + select)

	sub_datadir = ''
	if select not in ['all']:
		sub_datadir = data_path + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/'

	previous_datadir = ''
	if select not in ['0', 'all']:
		previous_datadir = data_path + '/' + size + '/' + method + '/' + select_interval + '/select' + str(int(select) - int(select_interval)) + '/'

	if select == '0':
		original_data = pd.read_csv(data_path + 'train.csv')
		path_list = original_data['path']
		transcript_list = original_data['transcript']
		dur_list = original_data['duration']
		speaker_list = original_data['speaker']
		combined_data = [list(tok) for tok in zip(path_list, transcript_list, dur_list, speaker_list)]
		shuffle(combined_data)
		
		initial_train_data = []
		initial_train_dur = 0
		select_data = []
		select_dur = 0

		for tok in combined_data:
			if initial_train_dur < 31 * 60:
				initial_train_data.append(tok)
				initial_train_dur += float(tok[-2])
			else:
				select_data.append(tok)
				select_dur += float(tok[-2])

		print('Initial training data size: ' + str(initial_train_dur / 60))
		print('Select data size: ' + str(select_dur / 60))

		initial_train_wav_path_list = [tok[0] for tok in initial_train_data]
		initial_train_transcript_list = [tok[1] for tok in initial_train_data]
		initial_train_dur_list = [tok[-2] for tok in initial_train_data]
		initial_train_speaker_list = [tok[-1] for tok in initial_train_data]

		initial_train_output = pd.DataFrame({'path': initial_train_wav_path_list,
				'transcript': initial_train_transcript_list,
				'duration': initial_train_dur_list,
				'speaker': initial_train_speaker_list})

		initial_train_output.to_csv(sub_datadir + 'train.' + size + '.input', index = False)

		select_wav_path_list = [tok[0] for tok in select_data]
		select_transcript_list = [tok[1] for tok in select_data]
		select_dur_list = [tok[-2] for tok in select_data]
		select_speaker_list = [tok[-1] for tok in select_data]

		select_output = pd.DataFrame({'path': select_wav_path_list,
				'transcript': select_transcript_list,
				'duration': select_dur_list,
				'speaker': select_speaker_list})

		select_output.to_csv(sub_datadir + 'select.' + size + '.input', index = False)

	if select not in ['0', 'all']:
		os.system('cat ' + previous_datadir + 'train.' + size + '.input ' + previous_datadir + '/increment.input >' + sub_datadir + 'train.' + size + '.input')
		os.system('mv ' + previous_datadir + 'residual.input ' + sub_datadir + 'select.' + size + '.input')

	print('loading data')
	train_data = ''
	if select not in ['all']:
		print(sub_datadir + '/train.' + size + '.input')
		train_data = get_data_reg(sub_datadir, 'train.' + size + '.input')
	else:
		print('Full training data')
		train_data = get_data_reg(data_path, 'train.csv')

	test_data = get_data_reg(data_path, 'test.csv')

	train(lang, data_path, size, select_interval, select, method, train_data, test_data, pretrained_model)

if __name__ == "__main__":
	main()
