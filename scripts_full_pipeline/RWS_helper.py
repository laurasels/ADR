import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
import os
from absl import app, flags, logging
from absl.flags import FLAGS

def custom_augmentation(train_dataset):
    '''
    Function that adds custom augmentation to dataset.
    Each line of augmentation occurs only in 25% of the cases
    
    Parameters
    ----------
    Tensorflow dataset with image and label column

    Returns
    -------
    Tensorflow dataset with augmentation
    '''
    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
        lambda: (tf.image.random_saturation(image,lower=0.3, upper=2.5),label),
        lambda: (image,label)))
    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
        lambda: (tf.image.random_contrast(image,lower=0.5, upper=2.0),label),
        lambda: (image,label)))
    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
        lambda: (tf.image.random_brightness(image,max_delta=0.15),label),
        lambda: (image,label)))
    train_dataset = train_dataset.map(lambda image, label: tf.cond(tf.random.uniform([], 0, 1) > 0.75,
        lambda: (tf.image.random_hue(image,max_delta=0.06),label),
        lambda: (image,label)))
    return train_dataset


def plot_images(dataset, n_images, samples_per_image, save_path):
    '''
    Function to plot amount of images after the augmentation
    
    Parameters
    ----------
    dataset             Tensorflow dataset
    n_images            How many images to save
    samples_per_image   How many images per batch (script uses 10)
    save_path           Location to save the images

    Returns
    -------
    n_images with data augmentation at specified location
    '''
    imagenr = 0
    for images in dataset.repeat(samples_per_image).batch(1):
        print(images[0].shape)
        if images[0].shape == (1, 16, 416, 416, 3):
            image = np.vstack(images[0][:,0,:,:,:].numpy())
            plt.figure()
            plt.imsave('%s/augmentation_%s.png' %(save_path,imagenr),image)
            plt.close()
            print('saving %s/augmentation_%s.png' %(save_path,imagenr))
            imagenr+=1
            if imagenr > (n_images-1):
                print('saving done')
                return

def broadcast_iou_eval(box_1, box_2):
    '''
    Calculate the iou for the evaluation 
    
    Parameters
    ----------
    Two bounding box coordinates [xmin,ymin,xmax,ymax]

    Returns
    -------
    IOU value of the boxes
    '''
    int_w = max(min(box_1[2], box_2[2]) - 
                max(box_1[0], box_2[0]), 0)
    int_h = max(min(box_1[3], box_2[3]) - 
                max(box_1[1], box_2[1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[2] - box_1[0]) * \
        (box_1[3] - box_1[1])
    box_2_area = (box_2[2] - box_2[0]) * \
        (box_2[3] - box_2[1])
    return np.array(int_area / (box_1_area + box_2_area - int_area))

def plot_training_curve():
    '''
    Plot the training curve in the main directory
    '''
    file = os.path.join(FLAGS.main_dir, 'training.log')

    df = pd.read_csv(file)
    val = df[['epoch','val_loss','val_yolo_output_0_loss','val_yolo_output_1_loss']]
    
    
    
    params = {'legend.fontsize': 'large',
              'figure.figsize': (12, 12),
             'axes.labelsize': 'large',
             'axes.titlesize':'x-large',
             'xtick.labelsize':'large',
             'ytick.labelsize':'large'}
    
    plt.rcParams.update(params)
    
    # Create figure
    fig = plt.figure(figsize=[10,5])
    gs = GridSpec(2,1,width_ratios=[1],height_ratios=[1,1]) # rows, columns, width per column, height per column
    
    # First subplot
    ax1 = fig.add_subplot(gs[0])
    
    df.plot(kind='line',x='epoch',y='loss',c='black',ax=ax1)
    df.plot(kind='line',x='epoch',y='val_loss',c='red',ax=ax1)
    ax1.set_ylim([0,val['val_loss'].min()*25])
    
    
    ax2 = fig.add_subplot(gs[1])
    
    val.plot(kind='line',x='epoch',y='val_yolo_output_0_loss',c='black',ax=ax2)
    val.plot(kind='line',x='epoch',y='val_yolo_output_1_loss',c='red',ax=ax2)
    
    ax2.set_ylim([0,10])
    plt.savefig(os.path.join(FLAGS.main_dir, 'training_loss.png'))

