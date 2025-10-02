import io, os
import pandas as pd
import statistics

sizes = ['30', '60']

for size in sizes:
	col_names =  ['Language', 'Task', 'Size', 'Select_interval', 'Select_size', 'Actual_duration', 'Model', 'Metric', 'Value', 'Method', 'Std']
	output  = pd.DataFrame(columns = col_names)

	ave_dur_list = []
	ave_value_list = []
	std_value_list = []
	file1 = pd.read_csv('asr_results_' + size + '_1.txt', sep = ' ')
	file2 = pd.read_csv('asr_results_' + size + '_2.txt', sep = ' ')
	file3 = pd.read_csv('asr_results_' + size + '_3.txt', sep = ' ')
	file1_dur_list = file1['Actual_duration'].tolist()
	file2_dur_list = file2['Actual_duration'].tolist()
	file3_dur_list = file3['Actual_duration'].tolist()
	file1_value_list = file1['Value'].tolist()
	file2_value_list = file2['Value'].tolist()
	file3_value_list = file3['Value'].tolist()

	output['Language'] = file1['Language'].tolist()
	output['Task'] = file1['Task'].tolist()
	output['Size'] = file1['Size'].tolist()
	output['Select_interval'] = file1['Select_interval'].tolist()
	output['Select_size'] = file1['Select_size'].tolist()
	output['Metric'] = file1['Metric'].tolist()
	output['Model'] = file1['Model'].tolist()
	output['Method'] = file1['Method'].tolist()

	for i in range(len(file1)):
		dur_list = [float(file1_dur_list[i]), float(file2_dur_list[i]), float(file3_dur_list[i])]
		value_list = [float(file1_value_list[i]), float(file2_value_list[i]), float(file2_value_list[i])]
		ave_dur_list.append(statistics.mean(dur_list))
		ave_value_list.append(statistics.mean(value_list))
		std_value_list.append(statistics.stdev(value_list))
		
	output['Actual_duration'] = ave_dur_list
	output['Value'] = ave_value_list
	output['Std'] = std_value_list

	output.to_csv('asr_ave_results_' + size + '.csv', index = False)