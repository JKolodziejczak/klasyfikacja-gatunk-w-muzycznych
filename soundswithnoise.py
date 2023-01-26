import math
import os
import csv

import librosa
import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write

snrs_to_test = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15]


def get_noise_from_sound(signal, noise, SNR):
    RMS_s = math.sqrt(np.mean(signal ** 2))
    # required RMS of noise
    RMS_n = math.sqrt(RMS_s ** 2 / (pow(10, SNR / 10)))

    # current RMS of noise
    RMS_n_current = math.sqrt(np.mean(noise ** 2))
    noise = noise * (RMS_n / RMS_n_current)

    return noise


def add_noise(dir):
    APP_FOLDER = 'data/genres_original/' + dir + "/"
    noise_file = 'data/noise_sample/pub3.wav'
    noise, sr = librosa.load(noise_file)
    noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))

    for base, dirs, files in os.walk(APP_FOLDER):
        for file in files:
            signal_file = base + file
            signal, sr = librosa.load(signal_file, duration=30.0)
            signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
            plt.plot(signal)
            plt.title(file)
            plt.xlabel("Sample number")
            plt.ylabel("Signal amplitude")
            # plt.show(block=False)
            # plt.pause(1)
            # plt.figure()

            # crop noise if its longer than signal

            if len(noise) > len(signal):
                noise = noise[0:len(signal)]
            if len(noise) < len(signal):
                signal = signal[0:len(noise)]

            noise = get_noise_from_sound(signal, noise, SNR=10)

            signal_noise = signal + noise

            plt.plot(signal_noise)
            plt.title(file + "with noise")
            plt.xlabel("Sample number")
            plt.ylabel("Amplitude")
            # plt.show(block=False)
            # plt.pause(1)
            # plt.figure()

            write("data/song_with_noise/" + dir + "/" + file, sr, signal_noise)


def save_to_csv(path):
    hop_length = 512
    n_fft = 2048  # FFT window size

    writer = create_csv_file_with_labels()

    for base, dirs, files in os.walk(path):
        for file in files:
            signal_file = base + "/" + file
            audio, sr = librosa.load(signal_file)
            chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sr,
                                                      n_fft=n_fft, hop_length=hop_length).flatten()
            rmse = librosa.feature.rms(y=audio, hop_length=hop_length).flatten()
            spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length).flatten()
            spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr,
                                                         n_fft=n_fft, hop_length=hop_length).flatten()
            rolloff = librosa.feature.spectral_rolloff(y=audio + 0.01, sr=sr,
                                                       n_fft=n_fft, hop_length=hop_length).flatten()
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length).flatten()

            harmony = librosa.feature.tempogram(audio, hop_length=hop_length).flatten()

            perceptrual = librosa.effects.percussive(audio).flatten()

            tempo = librosa.beat.tempo(y=audio, sr=sr, hop_length=hop_length)[0]

            length = len(audio)

            if length < 110000:
                continue

            splitted_path = base.split("/")
            genre = splitted_path[len(splitted_path) - 1]

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=hop_length)

            mfccs_features = np.hstack([
                np.mean(chroma_stft, axis=0),
                np.var(chroma_stft, axis=0),

                np.mean(rmse, axis=0),
                np.var(rmse, axis=0),

                np.mean(spec_cent, axis=0),
                np.var(spec_cent, axis=0),

                np.mean(spec_bw, axis=0),
                np.var(spec_bw, axis=0),

                np.mean(rolloff, axis=0),
                np.var(rolloff, axis=0),

                np.mean(zcr, axis=0),
                np.var(zcr, axis=0),

                np.mean(harmony, axis=0),
                np.var(harmony, axis=0),

                np.mean(perceptrual, axis=0),
                np.var(perceptrual, axis=0),

                tempo,

                np.mean(mfcc.T, axis=0),
                np.var(mfcc.T, axis=0),
            ])

            create_row_to_save(str(length), mfccs_features, genre, writer)


def create_row_to_save(length, mfccs_features, genre, writer):
    row = [length]
    for y in mfccs_features:
        row.append(y)
    row.append(genre)
    writer.writerow(row)


def create_csv_file_with_labels():
    labels = ["length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var",
              "spectral_centroid_mean",
              "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean",
              "rolloff_var",
              "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean",
              "perceptr_var",
              "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var", "mfcc4_mean",
              "mfcc4_var", "mfcc5_mean",
              "mfcc5_var", "mfcc6_mean", "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var",
              "mfcc9_mean", "mfcc9_var",
              "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean",
              "mfcc13_var",
              "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", "mfcc17_mean",
              "mfcc17_var",
              "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", "mfcc20_var", "label"]

    # open the file in the write mode
    f = open('data/features_5_sec_with_noise.csv', 'w', encoding='UTF8', newline='')
    # create the csv writer
    writer = csv.writer(f)
    writer.writerow(labels)
    return writer