import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def custom_augmentation(train_dataset):
    '''
    Function that adds custom augmentation to dataset.
    Each line of augmentation occurs only in 25% of the cases
    
    Input: tensorflow dataset with image and label column
    Output: tensorflow dataset with augmentation

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
        lambda: (tf.image.random_hue(image,max_delta=0.08),label),
        lambda: (image,label)))
    return train_dataset


def plot_images(dataset, n_images, samples_per_image,save_path):
    '''
    Function to plot amount of images after the augmentation
    '''
    imagenr = 0
    for images in dataset.repeat(samples_per_image).batch(1):
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
