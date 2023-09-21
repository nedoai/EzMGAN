latent_dim = 100 # noise vector. Highest vector - better sound

num_epochs = 10000

batch_size = 2

learning_rate = 0.0002 #lr

fixed_length = 160000 # 5 sec for example

samplerate = 16000 # Sr to load audios

gen_preview = 1000 # After each given number of epochs there will be a demonstration of the generator's operation in the form of an audio file

inf_about_each_epoch = 100 # Output of losses after a given number of passed epochs

audio_train_file = r"small_data/original_sound_-_laydshanks-QeqBfiLl.mp3" # Choice your audio path
