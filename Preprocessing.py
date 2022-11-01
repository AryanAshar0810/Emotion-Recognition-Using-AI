import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n
df = pd.read_csv(r"C:\Users\Pranav\Downloads\emotion folder/fer2013.csv")
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")
labels = []
usage = []
for i in df["emotion"]:
    labels.append(i)
for i in df["Usage"]:
    usage.append(i)
print(set(labels))
print(set(usage))
for i, j, k in tqdm(zip(df["emotion"], df["pixels"], df["Usage"])):
    if k == "Training":
        if not os.path.exists("train"):
            os.mkdir("train")
        if i == 0:
            if not os.path.exists("train/angry"):
                os.mkdir("train/angry")
        if i == 1:
            if not os.path.exists("train/disgusted"):
                os.mkdir("train/disgusted")
        if i == 2:
            if not os.path.exists("train/fearful"):
                os.mkdir("train/fearful")
        if i == 3:
            if not os.path.exists("train/happy"):
                os.mkdir("train/happy")
        if i == 4:
            if not os.path.exists("train/sad"):
                os.mkdir("train/sad")
        if i == 5:
            if not os.path.exists("train/surprised"):
                os.mkdir("train/surprised")
        if i == 6:
            if not os.path.exists("train/neutral"):
                os.mkdir("train/neutral")
    else:
        if not os.path.exists("test"):
            os.mkdir("test")
        if i == 0:
            if not os.path.exists("test/angry"):
                os.mkdir("test/angry")
        if i == 1:
            if not os.path.exists("test/disgusted"):
                os.mkdir("test/disgusted")
        if i == 2:
            if not os.path.exists("test/fearful"):
                os.mkdir("test/fearful")
        if i == 3:
            if not os.path.exists("test/happy"):
                os.mkdir("test/happy")
        if i == 4:
            if not os.path.exists("test/sad"):
                os.mkdir("test/sad")
        if i == 5:
            if not os.path.exists("test/surprised"):
                os.mkdir("test/surprised")
        if i == 6:
            if not os.path.exists("test/neutral"):
                os.mkdir("test/neutral")
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])
    img = Image.fromarray(mat)
    if i < 28709:
        if df['emotion'][i] == 0:
            img.save('train/angry/im' + str(angry) + '.png')
            angry += 1
        elif df['emotion'][i] == 1:
            img.save('train/disgusted/im' + str(disgusted) + '.png')
            disgusted += 1
        elif df['emotion'][i] == 2:
            img.save('train/fearful/im' + str(fearful) + '.png')
            fearful += 1
        elif df['emotion'][i] == 3:
            img.save('train/happy/im' + str(happy) + '.png')
            happy += 1
        elif df['emotion'][i] == 4:
            img.save('train/sad/im' + str(sad) + '.png')
            sad += 1
        elif df['emotion'][i] == 5:
            img.save('train/surprised/im' + str(surprised) + '.png')
            surprised += 1
        elif df['emotion'][i] == 6:
            img.save('train/neutral/im' + str(neutral) + '.png')
            neutral += 1
    else:
        if df['emotion'][i] == 0:
            img.save('test/angry/im' + str(angry_test) + '.png')
            angry_test += 1
        elif df['emotion'][i] == 1:
            img.save('test/disgusted/im' + str(disgusted_test) + '.png')
            disgusted_test += 1
        elif df['emotion'][i] == 2:
            img.save('test/fearful/im' + str(fearful_test) + '.png')
            fearful_test += 1
        elif df['emotion'][i] == 3:
            img.save('test/happy/im' + str(happy_test) + '.png')
            happy_test += 1
        elif df['emotion'][i] == 4:
            img.save('test/sad/im' + str(sad_test) + '.png')
            sad_test += 1
        elif df['emotion'][i] == 5:
            img.save('test/surprised/im' + str(surprised_test) + '.png')
            surprised_test += 1
        elif df['emotion'][i] == 6:
            img.save('test/neutral/im' + str(neutral_test) + '.png')
            neutral_test += 1
print("Done!")