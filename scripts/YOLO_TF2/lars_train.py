from absl import app, flags, logging
from absl.flags import FLAGS
import warnings
warnings.filterwarnings("ignore")

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
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset
import matplotlib.pyplot as plt
import sys

from lars_helper import custom_augmentation, plot_images

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')
flags.DEFINE_integer('nr_aug_imgs',0,'number of augmented images to print')
flags.DEFINE_string('aug_img_path','./','folder where the augmented images are stored')

#def plot_images(dataset, n_images, samples_per_image):
#    imagenr = 0
#    for images in dataset.repeat(samples_per_image).batch(n_images):
#        if images[0].shape == (1, 16, 416, 416, 3):
#            image = np.vstack(images[0][:,0,:,:,:].numpy())
##            print(image)
#            plt.figure()
#            plt.imsave('augmentation_images/augmentation_%s.png' %(imagenr),image)
#            plt.close()
#            print('saving nr %s' %(imagenr))
#            imagenr+=1
#            if imagenr >199:
#                sys.exit()
##            print(images[0].shape)
##            print(type(images[0]))
##            print(np.vstack(images[0][:,0,:,:,:].numpy()).shape)
##            output[:, row*416:(row+1)*416] = np.vstack(images[0][:,0,:,:,:].numpy())
##            row += 1
#
##    plt.figure()
##    plt.imshow(output)
##    plt.save('augmentation.png')
#    print('saving done')
def main(_argv):
    print(FLAGS.aug_img_path)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    else:
        train_dataset = dataset.load_fake_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    # Data augmentation
#    train_dataset = train_dataset.map(lambda image, label: (tf.image.random_contrast(image,lower=0.7, upper=1.3),label))
#    train_dataset = train_dataset.map(lambda image, label: (tf.image.random_brightness(image,max_delta=0.05),label))
#    train_dataset = train_dataset.map(lambda image, label: (tf.image.random_hue(image,max_delta=0.08),label))
#    train_dataset = train_dataset.map(lambda image, label: (tf.image.random_saturation(image,lower=100.6, upper=200.6),label))

#    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
#        lambda: (tf.image.random_saturation(image,lower=0.3, upper=2.5),label),
#        lambda: (image,label)))
#    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
#        lambda: (tf.image.random_contrast(image,lower=0.5, upper=2.0),label),
#        lambda: (image,label)))
#    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
#        lambda: (tf.image.random_brightness(image,max_delta=0.15),label),
#        lambda: (image,label)))
#    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
#        lambda: (tf.image.random_hue(image,max_delta=0.08),label),
#        lambda: (image,label)))
    train_dataset = custom_augmentation(train_dataset)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size))).repeat()
    train_dataset = train_dataset.map(lambda image, label: (tf.clip_by_value(image, 0., 1.0),label))
    
    # Print some figures to check the augmentation
    if FLAGS.nr_aug_imgs > 0:
        plot_images(train_dataset, n_images=FLAGS.nr_aug_imgs, samples_per_image=10,save_path=FLAGS.aug_img_path)
    # allow loading during calculations
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    else:
        val_dataset = dataset.load_fake_dataset()
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
        model_pretrained.load_weights(FLAGS.weights)

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
        model.load_weights(FLAGS.weights)
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
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train.tf',
                            verbose=1, save_weights_only=True,
                            save_best_only=True,
                            monitor='val_loss',
                            mode='min'),
            TensorBoard(log_dir='logs'),
            CSVLogger('training.log')
        ]

#        print(len(list(train_dataset)))

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            steps_per_epoch=18,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
