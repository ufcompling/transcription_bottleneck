import os
import soundfile as sf
from tqdm  import tqdm
import numpy as np
from multiprocessing import Pool
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", task="transcribe")

vowels = ["a","e", "i","o","u","y","ï","à","ê","ɒ","ɔ","ɛ","ɩ","ʋ",
          "â","ê","î","ô","á","é","í","ó","ö","ü","ā","ē","ī","ō","ū","ʉ"]
main_path = "/blue/liu.ying/word_making/data/"
def compute_ratio(lang):
    data = {}

    lang_path = f"{main_path}{lang}/split/rand/train/"
    words = []
    min = 0
    values = []
    for elt in tqdm(os.listdir(lang_path)):
        if ".wav" in elt:
            wav_path = lang_path+elt
            w, sr = sf.read(wav_path)
            dur = len(w)/sr
            if lang!="Cree":
                with open(lang_path+elt.replace(".wav", ".txt"), mode="r", encoding="utf-8") as tfile:
                    content = tfile.read()
            else:
                with open(lang_path+elt.replace(".wav", ".sro"), mode="r", encoding="utf-8") as tfile:
                    content = tfile.read().lower()
            subwords = tokenizer(content).input_ids
            words = words+subwords

            min = min+dur
            if min>=60:

                values.append(len(words))
                min=0
                words = []
    data[lang] = np.mean(values)
    print(lang, np.mean(values))


# compute_ratio("Bambara")
        
with Pool() as pool:
    pool.map(compute_ratio, [x for x in os.listdir(main_path)])
