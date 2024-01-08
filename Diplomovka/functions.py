
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import tensorflow as tf
import tensorflow_io as tfio

def plot_wave(clip):
    plt.figure(1)
    plt.plot(clip.audio[0])
    plt.show()
def calculate_spectogram(clip, n_fft = 2048, hop_lenght = 512):
    stft = librosa.core.stft(clip.audio[0], n_fft=n_fft,hop_length=hop_lenght)
    spectogram = np.abs(stft)
    return spectogram
def plot_spectogram(spectogram,sr,hop_lenght):
    librosa.display.specshow(spectogram, sr=sr,hop_length=hop_lenght)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.title("Spectogram for class: ")
    plt.colorbar()
    plt.show()
def preprocess(wav, label, sr, mels, slice_length):
    # Pad or trim the wav data
    wav = wav[:slice_length]
    padding_length = slice_length - len(wav)
    wav = np.pad(wav, (0, padding_length), mode='constant')

    # Convert to TensorFlow tensor for further processing
    wav_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)

    spectrogram = tfio.audio.spectrogram(wav_tensor, nfft=512, window=512, stride=256)
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=sr, mels=mels, fmin=0, fmax=20000)
    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)

    return dbscale_mel_spectrogram, label
def normalize_audio(audio_data):
    # Calculate the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(audio_data))
    
    # Check if the maximum amplitude is non-zero to avoid division by zero
    if max_amplitude > 0.0:
        # Normalize the audio data by dividing by the maximum amplitude
        normalized_audio = audio_data / max_amplitude
    else:
        # If all samples are zero, return the input unchanged to avoid division by zero
        normalized_audio = audio_data

    return normalized_audio