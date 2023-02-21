import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import librosa
import librosa.display
import warnings

warnings.filterwarnings('ignore')

from genreclassification import genre_classification, optimal_params
from soundswithnoise import save_to_csv, add_noise
from splitsounds import split_sounds

general_path = 'data'
print(list(os.listdir(f'{general_path}/genres_original/')))


def visualization():
    y, sr = librosa.load(f'{general_path}/genres_original/rock/rock.00030.wav')

    print('y:', y, '\n')
    print('y shape:', np.shape(y), '\n')
    print('Sample Rate (KHz):', sr, '\n')
    print('Audio length:', 661794 / 22050)

    # Delete silence before and after the actual audio
    audio_file, _ = librosa.effects.trim(y)

    print('Audio File:', audio_file, '\n')
    print('Audio File shape:', np.shape(audio_file))

    plt.figure(figsize=(16, 6))
    librosa.display.waveshow(y=audio_file, sr=sr, color="#A300F9");
    plt.title("Sound Waves in Rock sound", fontsize=23);

    n_fft = 2048  # FFT window size
    hop_length = 512  # number audio of frames between STFT columns

    # Short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(audio_file, n_fft=n_fft, hop_length=hop_length))
    print('Shape of D object:', np.shape(D))

    plt.figure(figsize=(16, 6))
    plt.plot(D);

    # Amplitude spectrogram to Decibels-scaled spectrogram.
    DB = librosa.amplitude_to_db(D, ref=np.max)

    # Spectogram
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log',
                             cmap='cool')
    plt.colorbar()
    plt.show()

    # Split harmonic and percussive
    y_harm, y_perc = librosa.effects.hpss(audio_file)

    plt.figure(figsize=(16, 6))
    plt.plot(y_harm, color='#A300F9');
    plt.plot(y_perc, color='#FFB100');

    # Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(audio_file, sr=sr)[0]
    print('Centroids:', spectral_centroids, '\n')
    print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')

    # Compute the time variable for visualization
    frames = range(len(spectral_centroids))
    # Convert frame counts to time (seconds)
    t = librosa.frames_to_time(frames)
    print('frames:', frames, '\n')
    print('t:', t)

    # Normalize the data
    def normalize(x, axis=0):
        return sklearn.preprocessing.minmax_scale(x, axis=axis)

    # Show spectral Centroid on plot
    plt.figure(figsize=(16, 6))
    plt.title("Spectral centroid", fontsize=23);
    librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color='#A300F9');
    plt.plot(t, normalize(spectral_centroids), color='#FFB100');

    # How granular you want your data to be
    hop_length = 5000

    # Chromogram
    chromagram = librosa.feature.chroma_stft(audio_file, sr=sr, hop_length=hop_length)
    print('Chromogram shape:', chromagram.shape)

    plt.figure(figsize=(16, 6))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
    plt.show()


if __name__ == '__main__':
    #visualization()
    #add_noise("blues")
    genre_classification()
    #split_sounds('rock')
    #save_to_csv('data/genres_original/')
    #save_to_csv('data/song_with_noise_5sec/')
    #test('rock')
    #optimal_params()

