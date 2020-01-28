import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs,broadcast_iou
from lars_helper import broadcast_iou_eval
import sys

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)

    
    false_positives = []
    false_negatives = []
    true_positives = []
    
    detected_annotations = []
    print('original label',_label.numpy())
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
    print('type of first item in labellist', type(labellist[0]))
    sublabellist = [item for item in labellist if item not in detected_annotations]
    print('undetected labels: ',sublabellist)
    for label in sublabellist:
        false_positives.append(0)
        false_negatives.append(1)
        true_positives.append(0)
        print('FN')

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
    sys.exit()
    img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))
    sys.exit()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
