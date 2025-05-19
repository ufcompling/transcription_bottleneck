from pydub import AudioSegment
import re
import os

os.system('mkdir ../asr_corpora/Mixtec/')
os.system('mkdir ../asr_corpora/Mixtec/wav/')
os.system('mkdir ../asr_corpora/Mixtec/txt/')

# Paths to the input files
audio_path = '../asr_resource/SLR89/Yoloxochitl-Mixtec-for-ASR/Sound-files-Narratives-for-ASR/'
transcripts_path = '../asr_resource/SLR89/Yoloxochitl-Mixtec-for-ASR/Transcriptions-for-ASR/Transcriber_513/'

transcripts_folders = os.listdir(transcripts_path)

for directory in os.listdir(audio_path):
    ## Finding the corresponding transcript folder for each audio folder
     ## In some of the audio folders, there might further sub audio folders; here trying to get a list of *.wav files for each transcript folder
    transcripts_dir = {}
    transcripts = []

    for folder in transcripts_folders:
        if folder.startswith(directory):
            num_transc_files = 0        
            for transc in os.listdir(os.path.join(transcripts_path, folder)):
                if transc.endswith('trs'):
                    transcripts_dir[transc] = transcripts_path + '/' + folder
                    transcripts.append(transc)
                    num_transc_files += 1

            if num_transc_files == 0:
                try:
                    for transc_folder in os.listdir(os.path.join(transcripts_path, folder)):
                        for transc in os.listdir(transcripts_path + '/' + folder + '/' + transc_folder):
                            if transc.endswith('trs'):
                                transcripts_dir[transc] = transcripts_path + '/' + folder + '/' + transc_folder
                                transcripts.append(transc)
                                num_transc_files += 1

                except:
                    print(f'No corresponding transcripts folder for {os.path.join(audio_path, directory)}')

    ## In some of the audio folders, there might further sub audio folders; here trying to get a list of *.wav files for each transcript folder
    if transcripts_dir != '':
        audio_file_lists = os.listdir(os.path.join(audio_path, directory))
        num_wav_files = 0
        for audio_file in audio_file_lists:
            if audio_file.endswith('.wav'):
                num_wav_files += 1
        if num_wav_files == 0:
            audio_file_lists = []
            for audio_folder in os.listdir(os.path.join(audio_path, directory)):
                try:
                    for audio_file in os.listdir(audio_path + '/' + directory + '/' + audio_folder):
                        if audio_file.endswith('.wav'):
                            audio_file_lists.append(audio_file)
                except:
                    print('No wav file in ' + audio_path + '/' + directory + '/' + audio_folder)

        for audio_file in audio_file_lists:
    #    for audio_file in os.listdir(os.path.join(audio_path, directory)):
            if audio_file.endswith('.wav'): # and 'Yolox_Agric_CTB501_Fases-de-la-luna-para-sembrar_2009-11-26-k' in audio_file:
                filename = audio_file.split('.')[0]
                gold_transcript = ''
                c = 0
                ## Finding the corresponding transcript for each audio file
                for transcript in transcripts:
                    if transcript.startswith(filename) and transcript.endswith('.trs'):
                        c += 1
                        gold_transcript = transcript
                try:
                    assert c == 1
                
                    # Parse the transcription and extract timestamps and text
                    utterances = []
                    current_text = []
                    current_start_time = 0

                    # Regex to capture <Sync time="..."/> tags and corresponding text
                    sync_pattern = re.compile(r'<Sync time="([\d.]+)"/>')

                    with open(os.path.join(transcripts_dir[gold_transcript], gold_transcript), 'r', encoding='iso-8859-1') as transc_file:
                        for line in transc_file:
                            sync_match = sync_pattern.search(line)
                    #        try:
                    #            print(sync_match, sync_match.group(1))
                    #        except:
                    #            pass
                            if sync_match:
                                # If we already have an active segment, save it
                                if current_text:
                                    utterances.append((current_start_time, float(sync_match.group(1)), " ".join(current_text)))
                                    current_text = []
                                # Update the start time for the next utterance
                                current_start_time = float(sync_match.group(1))
                            else:
                                # Collect text lines for the current segment
                                current_text.append(line.strip())

                    # Load the audio file
                    audio = AudioSegment.from_wav(os.path.join(audio_path, directory) + '/' + audio_file)

                    # Process and save each utterance as a separate audio and text file
                    for i, (start, end, text) in enumerate(utterances):
                        if end != 0.0:
                            # Convert time from seconds to milliseconds
                            start_ms = int(start * 1000)
                            end_ms = int(end * 1000)

                            # Extract audio segment
                            segment = audio[start_ms:end_ms]

                            # Save audio and transcript files
                            if not os.path.exists('../asr_corpora/Mixtec/wav/' + filename + '_' + str(i) + '.wav'):
                                segment.export('../asr_corpora/Mixtec/wav/' + filename + '_' + str(i) + '.wav', format="wav")
                        
                            if not os.path.exists('../asr_corpora/Mixtec/txt/' + filename + '_' + str(i) + '.txt'):
                                with open('../asr_corpora/Mixtec/txt/' + filename + '_' + str(i) + '.txt', "w", encoding="utf-8") as txt_file:
                                    txt_file.write(text + '\n')

                #    print(f"Extracted {len(utterances)} utterances to: ../asr_corpora/Mixtec/")

                except:
                    print(f"Multiple transcripts found for {filename}.wav in {transcripts_dir}")
            
                

#audio_path = '/mnt/data/Yolox_Agric_CTB501_Fases-de-la-luna-para-sembrar_2009-11-26-k.wav'
#transcription_path = '/mnt/data/Yolox_Agric_CTB501_Fases-de-la-luna-para-sembrar_2009-11-26-k_ed-2018-04-14.trs'
#output_dir = '/mnt/data/output_segments/'

# Ensure the output directory exists
#os.makedirs(output_dir, exist_ok=True)


