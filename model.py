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

BATCH_SIZE = 4
LAMBDA = 100
EPOCHS = 30
IMAGE_SIZE = 128


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


color_images, gray_images = prepare_data_from_dataframe(
    color_df_cleaned, gray_df_cleaned)

BATCH_SIZE = 4
LAMBDA = 100
EPOCHS = 30


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


def build_generator():
    """Build the U-Net generator model - true to pix2pix but with memory efficiency"""
    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])

    # Encoder (downsampling)
    # Using fewer filters at each layer compared to original pix2pix
    down_stack = [
        downsample(32, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 32)
        downsample(64, 4),                         # (batch_size, 32, 32, 64)
        downsample(128, 4),                        # (batch_size, 16, 16, 128)
        downsample(256, 4),                        # (batch_size, 8, 8, 256)
        downsample(512, 4),                        # (batch_size, 4, 4, 512)
        downsample(512, 4),                        # (batch_size, 2, 2, 512)
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
        upsample(512, 4, apply_dropout=True),      # (batch_size, 4, 4, 512)
        upsample(256, 4, apply_dropout=True),      # (batch_size, 8, 8, 256)
        upsample(128, 4),                          # (batch_size, 16, 16, 128)
        upsample(64, 4),                           # (batch_size, 32, 32, 64)
        upsample(32, 4),                           # (batch_size, 64, 64, 32)
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


def build_discriminator():
    """Build the PatchGAN discriminator - true to pix2pix but with memory efficiency"""
    initializer = tf.random_normal_initializer(0., 0.02)

    # Input layers
    inp = tf.keras.layers.Input(
        shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name='input_image')
    tar = tf.keras.layers.Input(
        shape=[IMAGE_SIZE, IMAGE_SIZE, 3], name='target_image')

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


@tf.function
def train_step(input_images, target_images, generator, discriminator,
               generator_optimizer, discriminator_optimizer):
    """Single training step"""
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate colorized output
        gen_output = generator(input_images, training=True)

        # Discriminator predictions
        disc_real_output = discriminator(
            [input_images, target_images], training=True)
        disc_generated_output = discriminator(
            [input_images, gen_output], training=True)

        # Calculate losses
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target_images)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    # Calculate gradients
    generator_gradients = gen_tape.gradient(
        gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


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
    plt.figure(figsize=(8, 2))
    
    # Add the sample name to the figure title if provided
    title_text = f"Image Colorization: {sample_name}" if sample_name else "Image Colorization Results"
    plt.suptitle(title_text, fontsize=14)

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


def train_model(train_dataset, val_dataset, train_filenames_dataset, val_filenames_dataset, epochs):
    """Train the pix2pix model"""
    print("Starting training...")
    # Build models
    generator = build_generator()
    print(generator.summary())
    discriminator = build_discriminator()
    print(discriminator.summary())

    # Define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(5e-6, beta_1=0.3)

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
        if (epoch + 1) % 2 == 0:
            # Now pass the filenames dataset to the display function
            display_sample_results(generator, val_dataset, val_filenames_dataset, num_samples=4)

    # Final save
    checkpoint.save(file_prefix=checkpoint_prefix)

    return generator, discriminator
