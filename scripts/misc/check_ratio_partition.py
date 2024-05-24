import os,re
from transformers import WhisperTokenizer
import numpy as np
from tqdm import tqdm

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\]\[]'
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", task="transcribe")

def clean_sent(sent):
    sent = re.sub(chars_to_remove_regex, '', sent).lower()
    return sent

main_path = "/blue/liu.ying/word_making/data/"

data = {}
for lang_fold in tqdm(os.listdir(main_path)):
    lang_path = main_path+lang_fold+"/split/"
    data[lang_fold] = {}
    for split in os.listdir(lang_path):
        data[lang_fold][split] = {"train" : {"tokens" : [], "subwords" : []}, "test" : {"tokens" : [], "subwords" : []}}
        
        for partition in ["train", "test"]:
            part_path = lang_path+split+f"/{partition}/"
            for txtfile in os.listdir(part_path):
                if ".txt" in txtfile or ".sro" in txtfile:
                    with open(part_path+txtfile, mode="r", encoding="utf-8") as tfile:
                        content=clean_sent(tfile.read())
                    data[lang_fold][split][partition]["tokens"] = data[lang_fold][split][partition]["tokens"]+[x for x in content.split()]
                    subwords = tokenizer(content).input_ids
                    data[lang_fold][split][partition]["subwords"] = data[lang_fold][split][partition]["subwords"]+subwords

import csv

with open("analyze_ratio.csv", mode="w", encoding="utf-8", newline='') as cfile:
    writer = csv.writer(cfile)
    
    # Write header
    header = ["lang"]
    for split in [1, 2, 3]:
        header.extend([
            f"tokens_{split}", f"subwords_{split}", 
            f"types_{split}", f"subtypes_{split}"
        ])
    writer.writerow(header)
    
    # Write data
    for lang in data:
        row = [lang]
        for split in data[lang]:
            tok_inter = [x for x in data[lang][split]["test"]["tokens"] if x not in data[lang][split]["train"]["tokens"]]
            sub_inter = [x for x in data[lang][split]["test"]["subwords"] if x not in data[lang][split]["train"]["subwords"]]
            ratio_tok = len(tok_inter)/len(data[lang][split]["test"]["tokens"])
            ratio_type = len(set(tok_inter))/len(set(data[lang][split]["test"]["tokens"]))
            ratio_type = f"{len(set(tok_inter))}/{len(set(data[lang][split]['test']['tokens']))}"
            # ratio_subtok = len(sub_inter)/len(data[lang][split]["test"]["subwords"])
            ratio_subtok = f"{len(sub_inter)}/{len(data[lang][split]['test']['subwords'])}"
            ratio_subtyp = len(set(sub_inter))/len(set(data[lang][split]["test"]["subwords"]))
            row.extend([ratio_tok, ratio_subtok, ratio_type, ratio_subtyp])
        writer.writerow(row)

