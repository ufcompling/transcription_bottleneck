import argparse
import os, re, sys
from transformers import Wav2Vec2Processor, AutoModelForCTC
import soundfile as sf
from datasets import load_metric
from jiwer import wer
import torch
from pyctcdecode import build_ctcdecoder
import numpy as np
from tqdm import tqdm
import shutil 

oov_rate = sys.argv[1]
lang = "Hupa"

lm_model= "/blue/liu.ying/word_making/data/{}/split/combined_{}.arpa".format(lang, oov_rate)
data_path = "/blue/liu.ying/word_making/data/{}/split/{}/test/".format(lang, oov_rate)
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�]'



lm = "no_lm"

# checkpoint = sorted([x for x in os.listdir("./xlsr53_{}/".format(lang+oov_rate)) if "checkpoint" in x], reverse=True)[0]
checkpoint = max([x for x in os.listdir("./xlsr53_{}/".format(lang+oov_rate)) if "checkpoint" in x], key=lambda y: int(y.split('-')[1]))

path_models = "./xlsr53_{}/".format(lang+oov_rate)
path_checkpoint = path_models+checkpoint

if "tokenizer_config.json" not in [x for x in os.listdir(path_checkpoint)]:
	shutil.copy(path_models+"tokenizer_config.json", path_checkpoint+"/tokenizer_config.json")
	shutil.copy(path_models+"vocab.json", path_checkpoint+"/vocab.json")

model = AutoModelForCTC.from_pretrained("./xlsr53_{}/{}/".format(lang+oov_rate, checkpoint)).to("cuda")#change



processor = Wav2Vec2Processor.from_pretrained("./xlsr53_{}/{}/".format(lang+oov_rate, checkpoint))
vocab = processor.tokenizer.get_vocab()


vocab[' '] = vocab['|']
del vocab[' ']
sorted_dict = {k.lower(): v for k, v in sorted(vocab.items(), key=lambda item: item[1])}
print(sorted_dict)

decoder = build_ctcdecoder(
	list(sorted_dict.keys()),
	lm_model,
	alpha = 0.5,
	beta = 1.5
)

data = []
long_wav = []
long_transc = ""

GS_idx = 0
print("load data")

for filename in os.listdir(data_path):
	if ".wav" in filename:
		
		with open(data_path+filename.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
			transc = tfile.read()
			transc = transc.replace("\n", " ")
			transc = re.sub(chars_to_remove_regex, '', transc).lower()
			# clean transcription
		if len(transc.split())>0:
			long_transc = long_transc+ " "+transc
			w, sr = sf.read(os.path.join(data_path+filename))
			try:
				long_wav = np.concatenate([long_wav, w])
			except:
				pass

			if (len(long_wav)/sr)>=5:#concatenetion of the corpus to have chunck of at least 23s
				entry = {}
				
				entry["sentence"] = long_transc
			
				entry["audio"] = {"sampling_rate" : sr, "array" : long_wav}
				data.append(entry)
				long_wav = []
				long_transc = ""

print("generate predictions")

signals = [x["audio"]["array"] for x in data]
sentences = [x["sentence"] for x in data]

preds = []

for i in tqdm(range(0, len(signals), 10)):
	sig = signals[i:i+10]
	inputs = processor(sig, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
	with torch.no_grad():
		logits = model(inputs).logits.to("cpu").numpy()
	decoded = []
	print(logits.shape)
	for ids in logits:

		beam_string = decoder.decode(ids).lower()
		decoded.append(beam_string)
		print(beam_string)
	
	preds = preds + decoded

#####ready for eval#####
with open("output_xlsr_{}{}.txt".format(lang, oov_rate), mode="w", encoding="utf-8") as tfile:
	for i in range(len(preds)):
		pred = preds[i].replace("\n", " ")
		ref = sentences[i].replace("\n", " ")
		tfile.write("prediction:  "+ pred+",")
		tfile.write("reference:   "+ ref+"\n")
		print("prediction:", pred)
		print("reference:", ref)
print(wer_metric.compute(predictions=preds, references=sentences))
print(cer_metric.compute(predictions=preds, references=sentences))

