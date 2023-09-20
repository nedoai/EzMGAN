import tensorflow as tf
import soundfile as sf
from keras.models import load_model
from cfg import latent_dim, batch_size

generator = load_model('generator.h5')

noise = tf.random.normal([batch_size, latent_dim])
generated_audio = generator.predict(noise)

output_audio_path = 'output_audio.wav'

sf.write(output_audio_path, generated_audio[0], 16000)
