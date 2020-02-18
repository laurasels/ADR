''' Only put images that have an annotation in the image directory
'''
import os
from pathlib import Path
from shutil import move, rmtree


from absl import app, flags, logging
from absl.flags import FLAGS

def match_annotated():
    """
    Move images if there is an annotation

    Parameters
    ----------
    Flags.main_dir         the main directory of the project
    Flags.img_type         extension of the image file    
    Returns
    -------
    Directory with images (raw/Images) that have an annotation in raw/Annotations
    """

    imginpath = os.path.join(FLAGS.main_dir, 'raw/')
    imgoutpath = os.path.join(FLAGS.main_dir, 'raw/Images/')
    xmlpath = os.path.join(FLAGS.main_dir, 'raw/Annotations')
    
    # Get an empty imgoutpath
    if os.path.exists(imgoutpath):
        rmtree(imgoutpath)
        os.makedirs(imgoutpath)
    else:
        os.makedirs(imgoutpath)

    # Move images if there is an annotation
    for f in os.listdir(xmlpath):
        if f.endswith('.xml'):
            print(f)
            f_out = f.replace('.xml',FLAGS.img_type)
            move(os.path.join(imginpath, f_out),os.path.join(imgoutpath, f_out))

