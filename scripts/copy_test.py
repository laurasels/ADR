import os,sys
from pathlib import Path
from shutil import copyfile,rmtree
import random
import numpy as np

path = Path('/computer-vision/ADR/data/testsetup')

imgindir = path / 'Images'
xmlindir = path / 'Annotations'

# remove old dirs if existing
for split in ['train','val']:
    if os.path.exists(path / split):
        print(path / split)
        rmtree(path / split)

# make new dirs
for split in ['train','val']:
    for mode in ['Images','Annotations']:
        os.makedirs(path / split / mode)


# list and shuffle all images
listfiles = os.listdir(str(path / 'Images'))
random.shuffle(listfiles)

# move files
counter=0
for file in listfiles:
    print(file)
    if counter <0.8*(len(listfiles)):
        split = 'train'
    else:
        split = 'val'
    print(counter,split)
    copyfile(imgindir / file,path / split / 'Images' / file)
    copyfile(xmlindir / file.replace('png','xml'),path / split / 'Annotations' / file.replace('png','xml'))
    counter+=1
