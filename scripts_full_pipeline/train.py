''' Train the Yolo model
'''
from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)
from Yolov3.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from Yolov3.utils import freeze_all
import Yolov3.dataset as dataset
import matplotlib.pyplot as plt
import sys
import os

from RWS_helper import custom_augmentation, plot_images, plot_training_curve

def train():
    '''
    Functon that deals with the training of the Yolo model

    Parameters
    ----------
    Flags.main_dir              Main directory of the project
    Flags.tiny                  Use Tiny or Full Yolo
    Flags.size                  size of the image (default 416)
    Flags.num_classes           Number of classes in the model 
    Flags.train_dataset         location of train (or val) dataset
    Flags.nr_aug_images         >0 means save some images after data augmentation to check the strength
    Flags.aug_img_path          path to the directory
    Flags.transfer              if transfer learning from darknet
    Flags.weights_num_classes   Number of classes in the Darknet model (=80)
    Flags.mode                  mode of tensorflow
    Flags.epochs                number of epochs for training
    Flags.model_save_dir        location to save the model

    Returns
    -------
    Saved model on the specified location
    training.log in the main directory
    '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
        model_loc = '/computer-vision/Yolov3_darknet_weights/yolov3-tiny.tf'
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
        model_loc = '/computer-vision/Yolov3_darknet_weights/yolov3.tf'

    if FLAGS.train_dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            os.path.join(FLAGS.main_dir, FLAGS.train_dataset),
            os.path.join(FLAGS.main_dir, FLAGS.classes), FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()
        #TODO exit with exitcode
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    
    # Data augmentation
    train_dataset = custom_augmentation(train_dataset)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size))).repeat()
    train_dataset = train_dataset.map(lambda image, label: (tf.clip_by_value(image, 0., 1.0),label))
    
    # Print some figures to check the augmentation
    if FLAGS.nr_aug_imgs > 0:
        aug_save_path = os.path.join(FLAGS.main_dir, FLAGS.aug_img_path)
        plot_images(train_dataset,
                n_images=FLAGS.nr_aug_imgs,
                samples_per_image=10,
                save_path=aug_save_path)
    # allow loading during calculations
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            os.path.join(FLAGS.main_dir, FLAGS.val_dataset),
            os.path.join(FLAGS.main_dir, FLAGS.classes), FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
        #TODO exit with exitcode

    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        model_pretrained.load_weights(model_loc)

        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(model_loc)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                patience=10, mode='min', cooldown=5, min_lr=0.0001, verbose=1,min_delta=0.05,),
            EarlyStopping(patience=30, verbose=1, monitor='val_loss', min_delta=0.001),
            ModelCheckpoint(os.path.join(FLAGS.main_dir, FLAGS.model_save_dir),
                            verbose=1, save_weights_only=True,
                            save_best_only=True,
                            monitor='val_loss',
                            mode='min'),
            TensorBoard(log_dir='logs'),
            CSVLogger(os.path.join(FLAGS.main_dir, 'training.log'))
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            steps_per_epoch=18,
                            callbacks=callbacks,
                            validation_data=val_dataset)


    plot_training_curve()
#if __name__ == '__main__':
#    try:
#        app.run(main)
#    except SystemExit:
       # pass
