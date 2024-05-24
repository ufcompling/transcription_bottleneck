import pandas as pd
import io, os

data_dir = 'data/Hupa/split/rand/'

wav_path_list = []
transcript_list = []

for split in ['train', 'test']: 
	for file in os.listdir(data_dir + split + '/'):
		if file.endswith('.wav'):
			wav_path = data_dir + split + '/' + file
			file_name = file.split('.')[0]
			transcipt_file = data_dir + split + '/' + file_name + '.txt'
			transcript = ''
			with open(transcipt_file) as f:
				for line in f:
					transcript = line.strip()

			wav_path_list.append(wav_path)
			transcript_list.append(transcript)

	output_data = pd.DataFrame({'path_to_the_wav_file': wav_path_list,
			  	'transcript': transcript_list})

	output_data.to_csv(data_dir + split + '.csv', index = False)