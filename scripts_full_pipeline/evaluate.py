''' Evaluation of the trained model
'''
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from Yolov3.models import (
    YoloV3, YoloV3Tiny
)
from Yolov3.dataset import transform_images, load_tfrecord_dataset
from Yolov3.utils import draw_outputs,broadcast_iou
from RWS_helper import broadcast_iou_eval
import sys
import os


def evaluate():
    '''
    Evaluation of the trained model

    Parameters
    ----------
    Flags.main_dir              Main directory of the project
    Flags.tiny                  Use Tiny or Full Yolo
    Flags.size                  size of the image (default 416)
    Flags.num_classes           Number of classes in the model 
    Flags.val_dataset           location of val dataset
    Flags.output                location to store the predicted images

    Returns
    -------
    All predicted images in the validation dataset
    Scores for the precision/recall
    '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(os.path.join(FLAGS.main_dir, FLAGS.model_save_dir)).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(os.path.join(FLAGS.main_dir, FLAGS.classes)).readlines()]
    logging.info('classes loaded')
    
    #tf.saved_model.save(yolo, os.path.join(FLAGS.main_dir, FLAGS.model_save_dir))

    # Initialize empty lists and print_counter
    false_positives = []
    false_negatives = []
    true_positives = []
    print_counter = 0

    # Load Dataset and loop over images, labels in the dataset
    dataset = load_tfrecord_dataset(
        os.path.join(FLAGS.main_dir, FLAGS.val_dataset),
        os.path.join(FLAGS.main_dir, FLAGS.classes), FLAGS.size)
#    dataset = dataset.shuffle(512)
    for img_raw, _label in dataset:
        labellist = [list(item) for item in list(_label.numpy()) if sum(item) > 0]
        print('labellist', labellist)
    
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, FLAGS.size)
    
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))
    
        logging.info('detections:')
        print(nums[0])
        
        detected_annotations = []
        for i in range(nums[0]):
            class_to_use = int(classes[0][i])
            score_to_use = np.array(scores[0][i])
            box_to_use = np.array(boxes[0][i])
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))
    
            # Filter labellist on class of detection and already detected labels
            sublabellist = [item for item in labellist if item[4] == class_to_use]
            print('detected_annotations', detected_annotations)
            for item in sublabellist:
                print('item in sublabellist after filtering', item)
            sublabellist = [item for item in sublabellist if item not in detected_annotations]
            # If no relevant labels --> FP
            if len(sublabellist) == 0:
                false_positives.append(1)
                false_negatives.append(0)
                true_positives.append(0)
                print('FP')
                continue
            
            # Determine iou for all relevant labels
            ious = []
            for label in sublabellist:
                ious.append(broadcast_iou_eval(box_to_use,label))
            
            # Select max iou and relevant label
            max_iou_place = np.argmax(ious)
            label = sublabellist[max_iou_place]
            max_iou = np.max(ious)
    
            if max_iou > 0.1 and label not in detected_annotations:
                detected_annotations.append(list(label))
                print('detected_annotations after appending', detected_annotations)
                false_positives.append(0)
                false_negatives.append(0)
                true_positives.append(1)
                print('TP - max iou = %s - confidence = %s' %(max_iou, score_to_use))
            else:
                false_positives.append(1)
                false_negatives.append(0)
                true_positives.append(0)
                print('FP')
    
        # Check for undetected labels --> FN
        sublabellist = [item for item in labellist if item not in detected_annotations]
        print('undetected labels: ',sublabellist)
        for label in sublabellist:
            false_positives.append(0)
            false_negatives.append(1)
            true_positives.append(0)
            print('FN')
    
        # Saving image for checking
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        outputloc = os.path.join(FLAGS.main_dir, FLAGS.output %(print_counter))
        cv2.imwrite(outputloc, img)
        logging.info('output saved to: {}'.format(outputloc))
        print_counter+=1
        
    # Final scores calculation    
    false_positives = np.sum(false_positives)
    false_negatives = np.sum(false_negatives)
    true_positives = np.sum(true_positives)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
                        
    print('FP: ', false_positives)
    print('FN: ', false_negatives)
    print('TP: ', true_positives)
    print('Precision: ', precision)
    print('Recall: ', recall)
