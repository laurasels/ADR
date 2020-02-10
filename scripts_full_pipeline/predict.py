import cv2
import tensorflow as tf
from Yolov3.models import (
    YoloV3, YoloV3Tiny
)
from Yolov3.dataset import transform_images
from Yolov3.utils import draw_outputs
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import numpy as np


def predict_helper(imgraw, yolo, outputloc, class_names):
    ''' Actual predict image and save the picture with the results on the specified location
    '''
    img = tf.expand_dims(imgraw, 0)
    img = transform_images(img, FLAGS.size)
    
    boxes, scores, classes, nums = yolo(img)
    
    img = cv2.cvtColor(imgraw, cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(outputloc, img)

def predict():
    ''' 
    Set up to predict image and save the picture with the results on the specified location

    Parameters
    ----------
    Flags.main_dir              Main directory of the project
    Flags.tiny                  Use Tiny or Full Yolo
    Flags.size                  size of the image (default 416)
    Flags.num_classes           Number of classes in the model
    Flags.predict_image         Path to image. Relative to main or absolute
    Flags.predict_folder        Path to image folder. Relative to main or absolute
    Flags.predict_output        Path to directory where the images will be saved

    Returns
    -------
    Predictions of specified image (folder) in the specified directory

    '''
    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(os.path.join(FLAGS.main_dir, FLAGS.model_save_dir)).expect_partial()
    class_names = [c.strip() for c in open(os.path.join(FLAGS.main_dir, FLAGS.classes)).readlines()]

    # Merge main dir to outputdir
    if FLAGS.predict_output.startswith('/'):
        outputloc = FLAGS.predict_output
    else:
        outputloc = os.path.join(FLAGS.main_dir, FLAGS.predict_output)
    
    if FLAGS.predict_image:
        if FLAGS.predict_image.startswith('/'):
            imgpath = FLAGS.predict_image
        else:
            imgpath = os.path.join(FLAGS.main_dir, FLAGS.predict_image)
        imgraw = cv2.imread(imgpath)

        # get last part of name and extension, insert '_predicted' and save on that location
        outputname = FLAGS.predict_image.split('/')[-1]
        outputextension = '.' + outputname.split('.')[-1]
        outputpath = os.path.join(outputloc,
                outputname.replace(outputextension,'_predicted'+outputextension))
        predict_helper(imgraw, yolo, outputpath, class_names)
    if FLAGS.predict_folder:
        if FLAGS.predict_folder.startswith('/'):
            imgpath = FLAGS.predict_folder
        else:
            imgpath = os.path.join(FLAGS.main_dir, FLAGS.predict_folder)
        for imgraw in os.listdir(imgpath):
            img = cv2.imread(os.path.join(imgpath, imgraw))
            # get last part of name and extension, insert '_predicted' and save on that location
            outputname = imgraw.split('/')[-1]
            outputextension = '.' + outputname.split('.')[-1]
            outputpath = os.path.join(outputloc,
                    outputname.replace(outputextension,'_predicted'+outputextension))
            predict_helper(img, yolo, outputpath, class_names)
