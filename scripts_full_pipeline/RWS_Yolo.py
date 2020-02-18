''' Module to train a Yolov3 model on a labeled image dataset
'''

from absl import app, flags, logging
from absl.flags import FLAGS

from coco2voc import coco2voc
from match_annotated import match_annotated
from make_train_test import maketraintest
from convert2tfrecord import convert2tfrecord
from train import train
from evaluate import evaluate
from predict import predict
import sys
import os

# TODO:
'''
- create classes.names
- include generate_anchors in train
- clean up print statements 
'''


# Required flags for some functionality:
flags.DEFINE_string('main_dir',None,'path to the main directory, other paths relative to this path')
flags.DEFINE_boolean('preprocessing',False,'preprocessing or not')
flags.DEFINE_boolean('train',False,'train or not')
flags.DEFINE_boolean('evaluate',False, 'evaluate or not')
flags.DEFINE_string('predict_output',None,'directory to store the predicted images. Either relative to main or absolute')

# Default option should work for all these following flags, specify if other option is wanted
# Base flags
flags.DEFINE_string('json_file','labels.json','path to the json file from Brainmatter')
flags.DEFINE_string('classes', 'classes.names', 'classes file')
flags.DEFINE_string('train_dataset', 'train/train.tfrecord', 'path to train_dataset')
flags.DEFINE_string('val_dataset', 'val/val.tfrecord', 'path to val_dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')

# training specific flags
flags.DEFINE_integer('nr_aug_imgs',0,'number of augmented images to print')
flags.DEFINE_string('aug_img_path','aug_ims','folder where the augmented images are stored')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'darknet',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 16, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', 80, 'specify num class for `darknet weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_string('model_save_dir', 'checkpoints/yolov3_train.tf', 'name of the final model weights')

# Predict or evaluate
flags.DEFINE_string('output', 'eval_images/output_%s.jpg', 'path to output image')
flags.DEFINE_float('yolo_iou_threshold', 0.1, 'iou threshold')
flags.DEFINE_float('yolo_score_threshold', 0.1, 'score threshold')
flags.DEFINE_string('predict_image',None,'image to predict. Either relative to main or absolute')
flags.DEFINE_string('predict_folder',None,'folder to predict. Either relative to main or absolute')

# Yolo darknet weights have hardcoded location:
# '/computer-vision/Yolov3_darknet_weights/yolov3.tf'


def main(_argv):
    """                                                                                                                                  
    Full pipeline to train a Yolo model on a labeled image dataset                                                                           
    Parameters
    ----------
    _argv =         Commandline arguments. The flags specified below.

    Necessary input:
    main_dir =      Location where all the project info will be stored
    labels =        labels of the relevant objects. Either in VOC format, or Coco format
    Images =        Directory with images, preferably 'main_dir/raw/Images'
    classes =       File with names of the classes (TODO: generate from labels/coco)
                                                                                                                                         
    Returns                                                                                                                              
    -------                                                                                                                              
    Depending on the flags:
    - trained model on the dataset with evaluation results
    - preprocessed directory structure to start training
    - augmented or predicted images
    """ 
    if FLAGS.preprocessing:
        # Conversion of the coco labels to the VOC labels
        coco2voc()
        # Move images if there is an annotation for it
        match_annotated()
        # Split the data in a train and test directory (80 - 20 split)
        maketraintest()
        # Convert to TFRecord dataset format
        for split in ['train','val']:
            convert2tfrecord(split)

    # Check if checkpoints directory exists
    checkpoint_path = os.path.join(FLAGS.main_dir, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print('new')
    else:
        print('exists')

    if FLAGS.train:
        # Actual training of the model
        train()
        # TODO generate anchors includen
    if FLAGS.evaluate:
        # Evaluation of the model, prints precision/recall etc
        evaluate()
    # Use model for prediction on images or folder
    if FLAGS.predict_output:
        predict()

if __name__ == '__main__':
    app.run(main)
