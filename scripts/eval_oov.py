import os
import sys

lang = "Hupa"
oov_rate = sys.argv[1]
#get all the types in the train set
types = set()
train_path = "/blue/liu.ying/word_making/data/{}/split/{}/train".format(lang,oov_rate)
for f in os.listdir(train_path):
    if ".txt" in f:
        with open(os.path.join(train_path, f), mode="r", encoding="utf-8") as tfile:
            transc = tfile.read()
        for i in transc.split():
            types.add(i)


#for each sentences in the test get the oov
#with open("output_{}{}.txt".format(lang,oov_rate), mode="r", encoding="utf-8") as tfile: # for whisper
with open("output_xlsr_{}{}.txt".format(lang,oov_rate), mode="r", encoding="utf-8") as tfile:
    output = tfile.read().split("\n")
ok_oov = 0
tot_oov = 0
for cpt,elt in enumerate(output):
    sp_elt = elt.split(",")
    if sp_elt[0]!="prediction" and len(sp_elt)>1:
        oov = [x for x in sp_elt[1].split() if x not in types]
        tot_oov+=len(oov)
        for o in oov:
            if o in sp_elt[0].split():
                ok_oov+=1
#for each oov in the sentence check if they exist in the pred
print(ok_oov/tot_oov)