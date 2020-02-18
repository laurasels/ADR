''' Split the dataset in train - test
'''
import os,sys
from pathlib import Path
from shutil import copyfile, rmtree
import random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS

# flags.DEFINE_string('main_dir','','path to main directory. train and val dirs will be created here')

def maketraintest():
    '''
    Split the dataset in train - test

    Parameters
    ----------
    Flags.main_dir         the main directory of the project
    Flags.img_type         extension of the image file

    Returns
    -------
    Two directories with the train/val dataset

    '''
    path = Path(FLAGS.main_dir)
    
    imgindir = path / 'raw/Images'
    xmlindir = path / 'raw/Annotations'
    
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
    listfiles = os.listdir(imgindir)
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
        copyfile(imgindir / file, path / split / 'Images' / file)
        copyfile(xmlindir / file.replace(FLAGS.image_type,'xml'), path / split / 'Annotations' / file.replace(FLAGS.image_type,'xml'))
        counter+=1

