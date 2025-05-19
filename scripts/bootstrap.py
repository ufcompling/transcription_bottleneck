### Bootstrap analysis ###

import io, os
import random
import pandas as pd
from transformers import Wav2Vec2Processor, AutoModelForCTC
from datasets import load_metric

wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

print('Loading test file')
test_file = pd.read_csv('data/Hupa/test.csv', sep = ',')
path_list = test_file['path'].tolist()
transcript_list = test_file['transcript'].tolist()
id_list = []
for i in range(len(path_list)):
	id_list.append(i)

### Collectin bootstrap samples
all_sample_ids = []
for i in range(10000):
	sample_ids = random.choices(id_list, k = len(id_list)) ### sampling with replacement
	all_sample_ids.append(sample_ids)

print(all_sample_ids[0])

output_file = open('bootstrap_results.txt', 'w')
header = ['Size', 'Select_size', 'Metric', 'Average', 'CI275', 'CI9750']
output_file.write('\t'.join(w for w in header) + '\n')

### Start bootstrap analysis
for size in ['30', '60']:
	for select_dir in os.listdir('data/Hupa/1/' + size + '/al/5/'):
		print(select_dir)
		select_size = select_dir[6 : ]
		pred_file = 'data/Hupa/1/' + size + '/al/5/' + select_dir + '/wav2vec2-large-xlsr-53_test_preds.txt'
		pred_list = []
		with open(pred_file) as f:
			for line in f:
				pred_list.append(line.strip())

		wer_list = []
		cer_list = []
		
		for sample_ids in all_sample_ids:
			transcript_sample = []
			pred_sample = []
			for idx in sample_ids:
				transcript_sample.append(transcript_list[idx])
				pred_sample.append(pred_list[idx])

			wer = wer_metric.compute(predictions=pred_sample, references=transcript_sample)
			cer = cer_metric.compute(predictions=pred_sample, references=transcript_sample)

			wer_list.append(wer)
			cer_list.append(cer)
		
		wer_list.sort()
		cer_list.sort()

		wer_ave = sum(wer_list) / len(wer_list)
		cer_ave = sum(cer_list) / len(cer_list)
		wer_ci_lower = wer_list[275]
		wer_ci_higher = wer_list[9750]
		cer_ci_lower = cer_list[275]
		cer_ci_higher = cer_list[9750]

		wer_info = [size, select_size, 'wer', wer_ave, wer_ci_lower, wer_ci_higher]
		cer_info = [size, select_size, 'cer', cer_ave, cer_ci_lower, cer_ci_higher]
		print(select_dir, wer_info, cer_info)
		output_file.write('\t'.join(str(w) for w in wer_info) + '\n')
		output_file.write('\t'.join(str(w) for w in cer_info) + '\n')


