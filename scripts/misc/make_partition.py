#!usr/bin/env python
# -*- coding: utf8 -*-
#
# -----------------------------------------------------------------------------
#   Created: 16/02/2024
#   Last Modified: 16/02/2024
# -----------------------------------------------------------------------------
#   Author: Éric Le Ferrand
#           Postdoctoral Researcher
#
#   Mail  : eleferrand@gmail.com / leferran@bc.edu
#  
#   Institution: Boston College
#
# ------------------------------------------------------------------------------
#   Description: 
#       Take a speech dataset a make partitions with different levels of Out Of
#       Vocabulary
# -----------------------------------------------------------------------------
lang = "Hupa"
values_oov= ["min", "max"]
margin = 0.01
text_path = "/blue/liu.ying/word_making/data/Hupa/txt/"
wav_path = "/blue/liu.ying/word_making/data/Hupa/wav/"

import os, re
import soundfile as sf
import random
import itertools
import numpy as np
from tqdm import tqdm
from inspect import getouterframes, currentframe

chars_to_remove_regex = '[\(\)\_\,\?\.\!\-\;\:\"\“\%\‘\”\�\]\[]'

def clean_sent(sent):
    sent = re.sub(chars_to_remove_regex, '', sent).lower()
    return sent

def swap_elements(train, test, value, out_train=[], out_test=[], max=float("-inf"), min=float("inf")):
    """
    Function that takes in input the train set, the test set and an OOV value
    This is a reccursive function that is going to sort both sets based on their number of oov for each utterance
    The function swap the elements in the test set that have the most frequent words with elements in the train set
    that have the least frequent words until the OOV rates reaches the value indicated (with a 0.02 margin).
    """
    #calculation of lexicon overlap
    
    set_test = set(list(itertools.chain.from_iterable([x["transc"].split() for x in test])))
    set_train = set(list(itertools.chain.from_iterable([x["transc"].split() for x in train])))
    oov_rate =len(set([x for x in set_test if x not in set_train]))/len(set_test)
    print('OOV rate: ' + str(oov_rate))
    print(len(set([x for x in set_test if x not in set_train])), len(set_test))
    print('\n')
    level = len(getouterframes(currentframe()))
    print(oov_rate, level)
    if level==700:
        print(oov_rate, type(out_train), type(out_test))
        train.clear()
        for x in out_train:
            train.append(x)
        test.clear()
        for x in out_test:
            test.append(x)
        return None
    if value=="max":#increase the number of oov
        for i in range(3):#I had an iteration here so more swapping can happen in one recursion
            swap_value = train.pop(0)
            test.insert(0, swap_value)
            
            swap_value = test.pop(-1)
            train.append(swap_value)
            if oov_rate>max:
                max = oov_rate
                out_train.clear()
                out_test.clear()
                for x in train:
                    out_train.append(x)
                for x in test:
                    out_test.append(x)

        swap_elements(train, test, value, out_train, out_test, max, min)

    elif value=="min":#decrease the number of oov
        swap_value = train.pop(-1)
        # test = test + [swap_value]
        test.append(swap_value)

        swap_value = test.pop(0)
        train.insert(0,swap_value)
        # train =  [swap_value] +train 
        if oov_rate<min and oov_rate>0:
            min = oov_rate
            out_train.clear()
            out_test.clear()
            for x in train:
                out_train.append(x)
            for x in test:
                out_test.append(x)
        
        swap_elements(train, test, value, out_train, out_test, max, min)


tokens = []
file_list = os.listdir(text_path)
random.shuffle(file_list)
total = 0
transc = []
audio = []
train = []
paths = []
test = []
freq = {}
for elt in tqdm(file_list):
    if ".txt" in elt:
        if elt.replace(".txt", ".wav") in os.listdir(wav_path):
            with open(text_path+elt, mode="r", encoding="utf-8") as tfile:
                content = clean_sent(tfile.read())
            if len(content.split())>0:
                transc.append(content)
                for w in content.split():
                    if w in freq:
                        freq[w] +=1
                    else:
                        freq[w] = 1
                signal = []
                sr = 16000
                signal, sr = sf.read(wav_path+elt.replace(".txt", ".wav"))
                paths.append(elt)
                
                dur = len(signal)/sr

                total += dur
                audio.append({"signal" : signal, "sr": sr, "dur": dur})
                wds = content.split()
                tokens = tokens+wds
            
cpt = 0
for ind in range(len(audio)):
    meansent = np.mean([freq[x] for x in transc[ind].split()])
    unique = len([wrd for wrd in transc[ind].split() if freq[wrd] in [1,2,3]])/len(transc[ind].split())
    
    if cpt< total*0.8: 
        train.append({"signal" : audio[ind]["signal"], "transc" : transc[ind].lower(),"path": paths[ind], "dur" : audio[ind]["dur"], "mean" : unique})
        cpt+=audio[ind]["dur"]
    else:
        test.append({"signal" : audio[ind]["signal"], "transc" : transc[ind].lower(),"path": paths[ind], "dur" : audio[ind]["dur"], "mean" : unique})


train.sort(key= lambda a: a["mean"], reverse=True)


test.sort(key= lambda a: a["mean"], reverse=True)

# train = train[len(train)//3:]
# test = test[:len(test)//3]

if not os.path.isdir("./data/{}/split/rand/".format(lang)):
    os.mkdir("./data/{}/split/rand/".format(lang))
    os.mkdir("./data/{}/split/rand/train/".format(lang))
    os.mkdir("./data/{}/split/rand/test/".format(lang))
for elt in train:
    src = text_path+elt["path"]
    dst = "./data/{}/split/rand/train/".format(lang)
    os.link(src, dst+os.path.basename(elt["path"]))
    src = wav_path+elt["path"]
    os.link(src.replace(".txt", ".wav"), dst+os.path.basename(elt["path"]).replace(".txt", ".wav"))

for elt in test:
    src = text_path+elt["path"]
    dst = "./data/{}/split/rand/test/".format(lang)
    os.link(src, dst+os.path.basename(elt["path"]))
    src = wav_path+elt["path"]
    os.link(src.replace(".txt", ".wav"), dst+os.path.basename(elt["path"]).replace(".txt", ".wav"))

for value in values_oov:
    print(sum(x["dur"] for x in test)/60)
    print(sum(x["dur"] for x in train)/60)

    print('\n')
    print(value)
    swap_elements(train, test, value)
    print(type(test[0]))

    print(sum(x["dur"] for x in test)/60)
    print(sum(x["dur"] for x in train)/60)
    print(len([x["path"] for x in train]))
    print(len(set([x["path"] for x in train])))

    if not os.path.isdir("./data/{}/split/{}/".format(lang, value)):
        os.mkdir("./data/{}/split/{}/".format(lang, value))
        os.mkdir("./data/{}/split/{}/train/".format(lang, value))
        os.mkdir("./data/{}/split/{}/test/".format(lang, value))
    for elt in train:
        src = text_path+elt["path"]
        dst = "./data/{}/split/{}/train/".format(lang, value)
        os.link(src, dst+os.path.basename(elt["path"]))
        src = wav_path+elt["path"]
        os.link(src.replace(".txt", ".wav"), dst+os.path.basename(elt["path"]).replace(".txt", ".wav"))

    for elt in test:
        src = text_path+elt["path"]
        dst = "./data/{}/split/{}/test/".format(lang, value)
        os.link(src, dst+os.path.basename(elt["path"]))
        src = wav_path+elt["path"]
        os.link(src.replace(".txt", ".wav"), dst+os.path.basename(elt["path"]).replace(".txt", ".wav"))


