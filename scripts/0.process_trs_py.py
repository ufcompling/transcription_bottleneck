# NOTE: this file is only tested to function in Python 3.12.10
# pydub doesn't work for me in 3.13
# pathlib.Path.walk() doesn't work for me in 3.10

import re
import string
from zipfile import ZipFile
from pathlib import Path
from pydub import AudioSegment


def unzip(dir_path: Path):
    """
        Unzips zipped files
        Skips zipped files if unzipped version exists
        (i.e., there exists a file of the same name)
    """
    print('Unzipping...')
    for root, dirs, files in dir_path.walk():
        for file in files:
            if file.endswith('.zip'):
                dir_path = root / file[:-4]  # make Path without .zip
                zip_path = root / file
                with ZipFile(zip_path, 'r') as zip_file:
                    try:  # Attempt to make directory
                        dir_path.mkdir()
                    except FileExistsError:  # If failed, already unzipped, so skip unzipping
                        continue
                    zip_file.extractall(dir_path)
                    print(f'Unzipped {zip_path.name}')
    print('Unzipping Complete!')


def get_wav_files(dir_path: Path):
    """
    Collects names of all wav files in dir_path
    Saves them in list
    """
    print('Gathering .WAV Files...')
    wav_files = []
    for root, dirs, files in dir_path.walk():
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(root / file[:-4])
    print('.WAV Files Gathered!')
    return wav_files


def get_tsr_data(tsr_file: Path):
    """
    Gathers relevant data from .TSR files
    Namely, audio_filename and transcript text and time
    """
    with open(tsr_file, 'r', encoding='ISO-8859-1') as tsr:
        text = tsr.read().splitlines()
        wav_file = re.search(r'audio_filename="([^\"]+)', text[2]).group(1)

        r = re.compile('<Sync.*')  # Find each <Sync> header. Keep its index and time
        syncs = {index: [re.search(r'time="([^\"]+)', line).group(1)]
                 for index, line in enumerate(text) if r.match(line)}

        for index in syncs.keys():
            i = 1
            line = ''
            while '<Sync' not in line and index + i <= len(text) - 1:
                line = text[index + i]  # gathers all utterances and matches them with sync index and time
                if not line.startswith('<') and line != '':
                    syncs[index].append(line)
                i += 1
    return wav_file, syncs


def create_txt_and_wav(wav_names: list, dir_path: Path, result_loc: Path):
    """
    Converts all trs files in dir_path & subdirs to txt files
    Places all txt files in a 'txt' folder at result_loc
    Only creates txt files that have matching wav files
    """
    print('Creating .TXT and .WAV Files...')
    txt_path = result_loc / 'txt'
    txt_path.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in dir_path.walk():
        for file in files:
            if file.endswith('.trs'):
                wav_name, syncs = get_tsr_data(root / file)
                wav_dir = ''.join([str(x) for x in wav_names if wav_name in str(x)])
                if wav_dir:  # ensures matching .wav file exists
                    i = 0
                    for item in syncs.items():  # joins multi-utterance syncs
                        text = ' '.join([x for x in item[1][1:]]).strip()
                        i += 1
                        file_name = txt_path / f'{wav_name}_part{i}'
                        with open(f'{file_name}.txt', 'w', encoding='utf-8') as out_file:
                            out_file.write(text)
                    wav_path = result_loc / 'wav'
                    wav_path.mkdir(parents=True, exist_ok=True)
                    create_wavs(wav_dir, syncs, result_loc)
    print('.TXT and .WAV Files Created!')


def create_wavs(wav_dir: string, syncs: dict, result_loc: Path):
    """
    Takes time stamp from syncs and cuts given .wav file into multiple smaller .wav files
    """
    print(f'{wav_dir}.wav')
    audio = AudioSegment.from_wav(f'{wav_dir}.wav')
    for i in range(len(syncs)):
        start_time = int(float(list(syncs.values())[i][0]) * 1000)  # find the start time in milliseconds
        end_time = int(float(list(syncs.values())[i + 1][0]) * 1000) if i + 1 < len(syncs) \
            else int(audio.duration_seconds * 1000)  # finds the next start time. it's the end time of previous
        audio_segment = audio[start_time:end_time]
        audio_name = str(Path(wav_dir).name)
        audio_segment.export(f'{result_loc}\\wav\\{audio_name}_part{i + 1}.wav', format='wav')
    pass


if __name__ == '__main__':
    ROOT_DIR = Path.cwd()
    RESULT_LOC = Path('..\\Nahuatl')

    unzip(ROOT_DIR)
    file_list = get_wav_files(ROOT_DIR)
    create_txt_and_wav(file_list, ROOT_DIR, RESULT_LOC)
