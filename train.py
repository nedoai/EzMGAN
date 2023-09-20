import tensorflow as tf
from keras import layers
import numpy as np
import os
import librosa
import soundfile as sf
from cfg import latent_dim, num_epochs, batch_size, learning_rate, fixed_length, gen_preview, inf_about_each_epoch, data_dir, samplerate

def load_and_preprocess_audio(file_path, target_sr=samplerate):
    audio, sr = librosa.load(file_path, sr=target_sr)
    
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    audio = audio / np.max(np.abs(audio))
    
    return audio

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(latent_dim,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(fixed_length, activation="tanh"))
    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(fixed_length,)))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy()

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

audio_files = os.listdir(data_dir)

for epoch in range(num_epochs):
    gen_loss_sum = 0.0
    disc_loss_sum = 0.0

    if len(audio_files) < batch_size:
        raise ValueError(f"Audio files smaller than the set batch_size\n\nAudios: {len(audio_files)}\nBatch_Size: {batch_size}")

    for _ in range(len(audio_files) // batch_size):
        batch_audio_data = []
        for _ in range(batch_size):
            audio_file = np.random.choice(audio_files) # Rand choice file. Idk why
            audio_path = os.path.join(data_dir, audio_file)
            audio, _ = librosa.load(audio_path, sr=samplerate)
            
            if len(audio) < fixed_length:
                audio = np.pad(audio, (0, fixed_length - len(audio)))
            else:
                audio = audio[:fixed_length]

            batch_audio_data.append(audio)
        batch_audio_data = np.array(batch_audio_data)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            noise = tf.random.normal([batch_size, latent_dim])
            generated_audio = generator(noise, training=True)
            real_output = discriminator(batch_audio_data, training=True)
            fake_output = discriminator(generated_audio, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        gen_loss_sum += gen_loss
        disc_loss_sum += disc_loss

    if (epoch + 1) % inf_about_each_epoch == 0:
        if (len(audio_files) // batch_size) > 0:
            gen_loss_avg = gen_loss_sum / (len(audio_files) // batch_size)
            disc_loss_avg = disc_loss_sum / (len(audio_files) // batch_size)
            print(f"Epoch {epoch + 1}/{num_epochs}, Gen Loss: {gen_loss_avg.numpy()}, Disc Loss: {disc_loss_avg.numpy()}")

    if (epoch + 1) % gen_preview == 0:
        num_samples = 1
        noise = tf.random.normal([num_samples, latent_dim])
        generated_audio = generator(noise, training=False).numpy()

        for i, audio_sample in enumerate(generated_audio):
            audio_path = f"generated_audiosample_{epoch}_{i}.wav"
            sf.write(audio_path, audio_sample, samplerate)

generator.save('generator.h5') # Or use .tf