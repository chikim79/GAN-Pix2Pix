# %% [markdown]
# # CSCA 5642 Final: Image Colorization using GAN

# %% [markdown]
# I will be tackling image colorization using GAN (Generative Adversarial Network).
# The dataset will be sourced from Kaggle's [Landscape color and grayscale images](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization)

# %% [markdown]
# ### Background
# 
# Generative Adversarial Network architecture consists of two primary components: a generator and a discriminator.  The generator attempts to create data that resembles a real data to "fool" the discriminator.  Discriminator on the other hand tries to distinguish between the real data and the data generator fabricated.  Generator's tasks is to produce results that are inditinguishable from real data.

# %% [markdown]
# ### Understanding different types of computer vision problems
# There are number of common computer vision and image manipulation problems.
# - Image-to-Image Translation
#   - Style Transfer: Converting photos to resemble paintings of specific artists
#   - Domain Translation: Converting horses to zebras, summer to winter scenes
#   - Colorization: Adding realistic color to grayscale images
#   - Super-Resolution: Generating high-resolution details from low-resolution images
#   - Image Inpainting: Filling in missing or damaged parts of images
#   - Sketch-to-Photo Conversion: Generating realistic images from simple sketches
#   - Image colorization provide a unique challenge that differ from other types of image manipulations.
# - Image Synthesis
#   - Photorealistic Image Generation: Creating completely new but realistic-looking images
#   - Conditional Image Generation: Creating images based on textual descriptions or class labels
#   - Face Generation: Creating realistic human faces that don't exist
#   - Texture Synthesis: Creating new textures with consistent patterns
# - Image Editing and Manipulation
#   - Face Aging/De-aging: Modifying images to show how someone might look older or younger
#   - Image Harmonization: Integrating edited content seamlessly into images
#   - Semantic Image Manipulation: Modifying images based on high-level descriptions
#   - Weather/Time of Day Manipulation: Changing the weather or lighting conditions in photos
# 
# We must choose an architecture that is a good fit for Image Colorization

# %% [markdown]
# ### Choosing the Ideal Machine Learning Model for Image Colorization
# 
# While there exists several options to train a deep learning model, after research, I have chosen pix2pix as the GAN architecture.
# 
# ### Other Options:
# 
# **Standard Autoencoders**
# 
# Just using autoencoders, instead of GAN, is the simplest approach to encode the grayscale images into compressed latent representation, then decoding this representation back into colorized image.  While this is straight forward, we would lose many details during the compression stage.
# 
# **Deep Convolutional GAN (DCGAN)**
# 
# I used DCGan for week 5 monet style image creation.  While DCGan is a good candidate to perform pure generation, we don't need to create any image from scratch,  we only need to keep the existing style, only change the color.
# 
# **CycleGAN**
# 
# CycleGan built for "unpaired" image-to-image translation.  This is a good fit style transfer. (Such as applying Monet style transfer on existing images.)  There is a one key difference, since this is "unpaired" there doesn't need to be a corresponding target output image for every input image in the dataset.  (such was the case for Modet style transfer problem)
# 
# ### Choosing Pix2Pix
# 
# What makes pix2pix particularly suited for image colorization is its ability to balance determinism (preserving the input structure) with creativity (applying colors). Also, we have target output image (colored image) for every input image (grayscale). Pix2Pix requires "paired" dataset, so the problem fits the Pix2Pix model the best.

# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# 

# %%
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import seaborn as sns
from collections import Counter
from PIL import Image
import pandas as pd


import tensorflow as tf
import time

# %%
IMAGE_PATH = './archive/landscape Images'

COLOR_PATH = IMAGE_PATH + '/color'
GRAY_PATH = IMAGE_PATH + '/gray'

# %%
print("Number of Color images: ",len(os.listdir(COLOR_PATH)))
print("Number of Gray images: ",len(os.listdir(GRAY_PATH)))

print("Color filenames: ",os.listdir(COLOR_PATH)[:20])
print("Gray filenames: ",os.listdir(GRAY_PATH)[:20])

NUMBER_OF_IMAGES = len(os.listdir(COLOR_PATH))

# %% [markdown]
# There are 7129 color and grayscale images.  Named from 0.jpg to 7128.jpg in each directory.

# %%
def create_dataframe(path):
    data = pd.DataFrame({'fileName': os.listdir(path)})
    data['filePath'] = data['fileName'].apply(lambda x: os.path.join(path, x))
    data['fileSize'] = data['filePath'].apply(lambda x: os.path.getsize(x))
    
    # to get around too many open files error
    # for (i, row) in data.iterrows():
    #     image = cv2.imread(row['filePath'])
    #     data.at[i, 'height'] = image.shape[0]
    #     data.at[i, 'width'] = image.shape[1]
    #     data.at[i, 'aspectRatio'] = round(image.shape[1] / image.shape[0], 2)
    #     data.at[i, 'channels'] = image.shape[2]
    data['img'] = data['filePath'].apply(lambda x: cv2.imread(x))
    data['height'] = data['img'].apply(lambda x: x.shape[0])
    data['width'] = data['img'].apply(lambda x: x.shape[1])
    data['aspectRatio'] = round(data['width'] / data['height'],2)
    data['channels'] = data['img'].apply(lambda x: x.shape[2])
    data['color'] = data['img'].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    data['gray'] = data['img'].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY))
    
    
        
    return data
    
color_df = create_dataframe(COLOR_PATH)
gray_df = create_dataframe(GRAY_PATH)

# %%
print("Color Image Info")
print(color_df.info())
print(color_df.describe())

# %%
print("Gray Image Info")
print(gray_df.info())
print(gray_df.describe())

# %% [markdown]
# Most images are 150x150.  There seems to be an outlier.  we'll remove this to get around any unwanted outlier

# %%
print("non 150 height images")
print(color_df[color_df['height'] != 150].shape)
print(gray_df[gray_df['height'] != 150].shape)

# %% [markdown]
# Only 23 images are not 150x150.  Let's make sure they are the same images, and remove them from the dataset

# %%
color_bad = color_df[color_df['height'] != 150].fileName
gray_bad = gray_df[gray_df['height'] != 150].fileName

print("Making sure the bad images are the same")
print(color_bad.equals(gray_bad))
print("Removing bad images")
color_df_cleaned = color_df[color_df['height'] == 150]
gray_df_cleaned = gray_df[gray_df['height'] == 150]

# %% [markdown]
# ## Sample Images

# %%
def display_sample_images(n=4, cols=4):
    
    sample_size = n * cols
    
    file_nums = np.random.randint(0, NUMBER_OF_IMAGES, sample_size // 2)
    
    plt.figure(figsize=(8, 8))
    for i, file_num in enumerate(file_nums):
        # Show Color Image
        img = Image.open(os.path.join(COLOR_PATH, str(file_num) + ".jpg"))
        plt.subplot(n, cols, i*2+1)
        plt.imshow(img)
        plt.axis('off')
        
        # Show Gray Image
        img2 = Image.open(os.path.join(GRAY_PATH, str(file_num) + ".jpg"))
        plt.subplot(n, cols, i*2+2)
        # color map gray.  Otherwise it will be shown in RGB
        plt.imshow(img2, cmap="gray")
        plt.axis('off')
    plt.show()
    
display_sample_images()


# %%
def analyze_color_distribution(images, n_samples=10):
    """
    Analyze the color distribution in the dataset
    """
    # Sample a subset of images for this analysis
    if len(images) > n_samples:
        sample_indices = np.random.choice(len(images), n_samples, replace=False)
        sample_images = [images[i] for i in sample_indices]
    else:
        sample_images = images
    
    # Analyze RGB distribution
    r_values = []
    g_values = []
    b_values = []
    
    for img in sample_images:
        # Flatten the image and collect color values
        pixels = img.reshape(-1, 3)
        r_values.extend(pixels[:, 0])
        g_values.extend(pixels[:, 1])
        b_values.extend(pixels[:, 2])
    
    # Plot histograms
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(r_values, bins=50, color='red', alpha=0.7)
    plt.title('Red Channel Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 2)
    plt.hist(g_values, bins=50, color='green', alpha=0.7)
    plt.title('Green Channel Distribution')
    plt.xlabel('Pixel Value')
    
    plt.subplot(1, 3, 3)
    plt.hist(b_values, bins=50, color='blue', alpha=0.7)
    plt.title('Blue Channel Distribution')
    plt.xlabel('Pixel Value')
    
    plt.tight_layout()
    plt.show()
    
    # return r_values, g_values, b_values



# %% [markdown]
# ## Color Images color distributions

# %%
analyze_color_distribution(color_df_cleaned['img'].values, n_samples=100)

# %% [markdown]
# ## Gray Images Color Distributions

# %%
analyze_color_distribution(gray_df_cleaned['img'].values, n_samples=100)

# %%
# Function to analyze color vs grayscale pixel correlation
def analyze_color_grayscale_correlation(color_images, gray_images, n_samples=5):
    """
    Analyze how well grayscale values correlate with original color values
    """
    if len(color_images) > n_samples:
        sample_indices = np.random.choice(len(color_images), n_samples, replace=False)
    else:
        sample_indices = range(len(color_images))
    
    correlations = []
    
    for idx in sample_indices:
        color_img = color_images[idx]
        gray_img = gray_images[idx]
        
        # Convert color image to grayscale for comparison
        color_gray = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
        
        # Extract first channel from grayscale image (all channels should be the same)
        if len(gray_img.shape) == 3:
            gray_channel = gray_img[:,:,0]
        else:
            gray_channel = gray_img
        
        # Calculate correlation
        correlation = np.corrcoef(color_gray.flatten(), gray_channel.flatten())[0,1]
        correlations.append(correlation)
        
        # Display example
        plt.figure(figsize=(6, 3))
        
        plt.subplot(1, 3, 1)
        plt.imshow(color_img)
        plt.title('Original Color')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(color_gray, cmap='gray')
        plt.title('Grayscale from Color')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(gray_channel, cmap='gray')
        plt.title('Dataset Grayscale')
        plt.axis('off')
        
        plt.suptitle(f'Correlation: {correlation:.4f}')
        plt.tight_layout()
        plt.show()
    
    return correlations

# %%
analyze_color_grayscale_correlation(color_df_cleaned['img'].values, gray_df_cleaned['img'].values, n_samples=10)

# %% [markdown]
# As expected, the color images turned to grayscale is pretty much the same as the gray images.

# %%
IMAGE_SIZE = 128

# %%
def prepare_data_from_dataframe(color_df, gray_df):
    """
    Prepare data from pandas DataFrames containing image data

    Args:
        color_df: DataFrame with color images
        gray_df: DataFrame with grayscale images

    Returns:
        Arrays of color and grayscale images, plus list of filenames
    """
    # Make sure the DataFrames are matched by filename
    color_df = color_df.set_index('fileName')
    gray_df = gray_df.set_index('fileName')
    common_files = set(color_df.index).intersection(set(gray_df.index))

    print(f"Found {len(common_files)} matching image pairs")

    # Create empty lists to store preprocessed images
    color_img = []
    gray_img = []
    filenames = []  # Track filenames

    # Process each image pair
    print("Processing images...")
    for filename in common_files:
        # Get color image and normalize
        color_image = color_df.loc[filename, 'color']

        # Get grayscale image and normalize
        gray_image = gray_df.loc[filename, 'color']

        color_image = cv2.resize(color_image, (IMAGE_SIZE, IMAGE_SIZE))
        gray_image = cv2.resize(gray_image, (IMAGE_SIZE, IMAGE_SIZE))

        color_image = (color_image.astype(np.float32) / 127.5) - 1.0
        gray_image = (gray_image.astype(np.float32) / 127.5) - 1.0

        color_img.append(color_image)
        gray_img.append(gray_image)
        filenames.append(filename)  # Store the filename

    return np.array(color_img), np.array(gray_img), filenames

color_images, gray_images, filenames = prepare_data_from_dataframe(color_df_cleaned, gray_df_cleaned)

# %%
BATCH_SIZE = 4
LAMBDA = 100
EPOCHS = 30

# %%
def preprocess_data(color_images, gray_images, filenames, train_split=0.8):
    """Split data into training and validation sets"""
    # Determine split index
    num_samples = len(color_images)
    split_idx = int(num_samples * train_split)

    # Create training datasets
    train_color = color_images[:split_idx]
    train_gray = gray_images[:split_idx]
    train_filenames = filenames[:split_idx]

    # Create validation datasets
    val_color = color_images[split_idx:]
    val_gray = gray_images[split_idx:]
    val_filenames = filenames[split_idx:]

    # Create TensorFlow datasets for images
    train_dataset = tf.data.Dataset.from_tensor_slices((train_gray, train_color))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_gray, val_color))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Create TensorFlow datasets for filenames
    train_filenames_dataset = tf.data.Dataset.from_tensor_slices(train_filenames)
    train_filenames_dataset = train_filenames_dataset.batch(BATCH_SIZE)

    val_filenames_dataset = tf.data.Dataset.from_tensor_slices(val_filenames)
    val_filenames_dataset = val_filenames_dataset.batch(BATCH_SIZE)

    return train_dataset, val_dataset, train_filenames_dataset, val_filenames_dataset

train_dataset, val_dataset, train_filenames_dataset, val_filenames_dataset = preprocess_data(
        color_images, gray_images, filenames)

# %%
# Model building functions
def downsample(filters, size, apply_batchnorm=True):
    """Downsampling block for the generator"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                              kernel_initializer=initializer, use_bias=False))
    
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    
    result.add(tf.keras.layers.LeakyReLU())
    
    return result

def upsample(filters, size, apply_dropout=False):
    """Upsampling block for the generator"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                       kernel_initializer=initializer, use_bias=False))
    
    result.add(tf.keras.layers.BatchNormalization())
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())
    
    return result

# %% [markdown]
# ## Building the U-Net Generator

# %%
def build_generator():
    """Build the U-Net generator model - true to pix2pix but with memory efficiency"""
    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    
    # Encoder (downsampling)
    # Using fewer filters at each layer compared to original pix2pix
    down_stack = [
        downsample(16, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 16)
        downsample(32, 4),                         # (batch_size, 32, 32, 32)
        downsample(64, 4),                         # (batch_size, 16, 16, 64)
        downsample(128, 4),                        # (batch_size, 8, 8, 128)
        downsample(256, 4),                        # (batch_size, 4, 4, 256)
        downsample(256, 4),                        # (batch_size, 2, 2, 256)
    ]
    
    # Encoder pass
    x = inputs
    skips = []
    
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Decoder (upsampling)
    # Using fewer filters at each layer compared to original pix2pix
    up_stack = [
        upsample(256, 4, apply_dropout=True),      # (batch_size, 4, 4, 256)
        upsample(128, 4, apply_dropout=True),      # (batch_size, 8, 8, 128)
        upsample(64, 4),                           # (batch_size, 16, 16, 64)
        upsample(32, 4),                           # (batch_size, 32, 32, 32)
        upsample(16, 4),                           # (batch_size, 64, 64, 16)
    ]
    
    # Decoder pass with skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # Handle dimension issues with resize if needed
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            x = tf.image.resize(x, [skip.shape[1], skip.shape[2]])
        x = tf.keras.layers.Concatenate()([x, skip])
    
    # Final output layer
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        3, 4, strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh')  # (batch_size, 128, 128, 3)
    
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

# %%
def build_discriminator():
    """Build the PatchGAN discriminator - true to pix2pix but with memory efficiency"""
    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Input layers
    inp = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name='target_image')
    
    # Concatenate inputs
    x = tf.keras.layers.Concatenate()([inp, tar])  # (batch_size, 128, 128, 6)
    
    # PatchGAN layers - using fewer filters than original
    down1 = downsample(16, 4, False)(x)            # (batch_size, 64, 64, 16)
    down2 = downsample(32, 4)(down1)               # (batch_size, 32, 32, 32)
    down3 = downsample(64, 4)(down2)               # (batch_size, 16, 16, 64)
    
    # Zero padding and final convolution
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
    conv = tf.keras.layers.Conv2D(
        256, 4, strides=1, padding='valid',
        kernel_initializer=initializer, use_bias=False)(zero_pad1)
    
    batchnorm = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)
    # Final 1x1 output (without sigmoid - will use from_logits=True in loss)
    last = tf.keras.layers.Conv2D(
        1, 4, strides=1, padding='valid',
        kernel_initializer=initializer)(zero_pad2)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=last)

# %%
def generator_loss(disc_generated_output, gen_output, target):
    """Calculate generator loss"""
    # GAN loss for fooling the discriminator
    gan_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(disc_generated_output), disc_generated_output)
    
    # L1 loss for color accuracy
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    
    # Combined loss
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    """Calculate discriminator loss"""
    # Real image loss
    real_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(disc_real_output), disc_real_output)
    
    # Generated image loss
    generated_loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)(tf.zeros_like(disc_generated_output), disc_generated_output)
    
    # Total loss
    total_disc_loss = real_loss + generated_loss
    
    return total_disc_loss

# %%
@tf.function
def train_step(input_images, target_images, generator, discriminator, 
               generator_optimizer, discriminator_optimizer):
    """Single training step"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate colorized output
        gen_output = generator(input_images, training=True)
        
        # Discriminator predictions
        disc_real_output = discriminator([input_images, target_images], training=True)
        disc_generated_output = discriminator([input_images, gen_output], training=True)
        
        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target_images)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    
    # Calculate gradients
    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss

# %%
def display_sample_results(model, dataset, filenames_dataset=None, num_samples=4):
    """Display sample results during training
    
    Args:
        model: The generator model
        dataset: Dataset containing input and target images
        filenames_dataset: Optional dataset containing filenames corresponding to the images
        num_samples: Number of samples to display
    """
    # Only take one batch
    if filenames_dataset is not None:
        # Get the filenames batch
        filenames_batch = next(iter(filenames_dataset.take(1)))
    else:
        filenames_batch = None
        
    # Get the image batch
    for input_image, target_image in dataset.take(1):
        # Ensure we don't try to display more samples than are in the batch
        samples_to_display = min(num_samples, input_image.shape[0])
        
        for i in range(samples_to_display):
            # Get the filename if available, otherwise use sample number
            if filenames_batch is not None and i < len(filenames_batch):
                filename = filenames_batch[i].numpy().decode('utf-8')
            else:
                filename = f"Sample {i+1}"
                
            generate_and_save_images(
                model,
                input_image[i:i+1],
                target_image[i:i+1],
                sample_name=filename
            )


def generate_and_save_images(model, test_input, target, filename=None, sample_name=None):
    """Generate colorized images and save if filename is provided
    
    Args:
        model: The generator model
        test_input: Input grayscale image
        target: Target color image
        filename: Optional filename to save the figure
        sample_name: Original filename or identifier to display in the title
    """
    prediction = model(test_input, training=False)

    # Adjusted figure size to better fit 128x128 images
    plt.figure(figsize=(7, 2))
    
    # Add the sample name to the figure title if provided
    title_text = f"Image Colorization: {sample_name}" if sample_name else "Image Colorization Results"
    plt.suptitle(title_text, fontsize=10)

    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input (Grayscale)', 'Ground Truth (Color)', 'Predicted (Color)']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])

        # Rescale to [0, 1] for display
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# %%
def train_model(train_dataset, val_dataset, train_filenames_dataset, val_filenames_dataset, epochs):
    """Train the pix2pix model"""
    print("Starting training...")
    # Build models
    generator = build_generator()
    print(generator.summary())
    discriminator = build_discriminator()
    print(discriminator.summary())

    # Define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(8e-5, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(3e-5, beta_1=0.3)

    # Create checkpoints
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator)

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()

        # Initialize metrics
        gen_total_loss_avg = tf.keras.metrics.Mean()
        gen_gan_loss_avg = tf.keras.metrics.Mean()
        gen_l1_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()

        print(f"Epoch {epoch+1}/{epochs}")

        # Training
        for input_image, target_image in train_dataset:
            # Train on batch
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(
                input_image, target_image, generator, discriminator,
                generator_optimizer, discriminator_optimizer)

            # Update metrics
            gen_total_loss_avg.update_state(gen_total_loss)
            gen_gan_loss_avg.update_state(gen_gan_loss)
            gen_l1_loss_avg.update_state(gen_l1_loss)
            disc_loss_avg.update_state(disc_loss)

        # Validation
        val_gen_total_loss_avg = tf.keras.metrics.Mean()
        val_gen_gan_loss_avg = tf.keras.metrics.Mean()
        val_gen_l1_loss_avg = tf.keras.metrics.Mean()
        val_disc_loss_avg = tf.keras.metrics.Mean()

        for input_image, target_image in val_dataset:
            # Generate output
            gen_output = generator(input_image, training=False)

            # Discriminator predictions
            disc_real_output = discriminator(
                [input_image, target_image], training=False)
            disc_generated_output = discriminator(
                [input_image, gen_output], training=False)

            # Calculate losses
            gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
                disc_generated_output, gen_output, target_image)
            disc_loss = discriminator_loss(
                disc_real_output, disc_generated_output)

            # Update metrics
            val_gen_total_loss_avg.update_state(gen_total_loss)
            val_gen_gan_loss_avg.update_state(gen_gan_loss)
            val_gen_l1_loss_avg.update_state(gen_l1_loss)
            val_disc_loss_avg.update_state(disc_loss)

        # Print epoch results
        time_taken = time.time() - start_time
        print(f"Time taken: {time_taken:.2f}s")

        print(f"Training: Gen Loss: {gen_total_loss_avg.result():.4f}, "
              f"Gen GAN Loss: {gen_gan_loss_avg.result():.4f}, "
              f"Gen L1 Loss: {gen_l1_loss_avg.result():.4f}, "
              f"Disc Loss: {disc_loss_avg.result():.4f}")

        print(f"Validation: Gen Loss: {val_gen_total_loss_avg.result():.4f}, "
              f"Gen GAN Loss: {val_gen_gan_loss_avg.result():.4f}, "
              f"Gen L1 Loss: {val_gen_l1_loss_avg.result():.4f}, "
              f"Disc Loss: {val_disc_loss_avg.result():.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        # Generate and display sample images
        if (epoch + 1) % 4 == 0:
            # Now pass the filenames dataset to the display function
            display_sample_results(generator, val_dataset, val_filenames_dataset, num_samples=4)

    # Final save
    checkpoint.save(file_prefix=checkpoint_prefix)

    return generator, discriminator

# %%
generator, discriminator = train_model(
        train_dataset, val_dataset, train_filenames_dataset, val_filenames_dataset, epochs=50)

# %%
import gc

def clear_memory():
    """Clear unused memory aggressively"""
    gc.collect()
    tf.keras.backend.clear_session()
    
    # More aggressive memory clearing in TensorFlow
    for device in tf.config.experimental.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats(device)
        except:
            pass
            
    # Sleep briefly to allow memory release
    import time
    time.sleep(0.5)
    
clear_memory()
    


# %%


# %%



