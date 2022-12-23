import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf


def split_sounds(dir):
    audio_dir = 'data/song_with_noise/'
    out_dir = 'data/song_with_noise_5sec/' + dir
    os.makedirs(out_dir, exist_ok=True)

    for base, dirs, files in os.walk(audio_dir+"/"+dir):
        for file in files:
            audio_file = os.path.join(audio_dir+"/"+dir, file)

            wave, sr = librosa.load(audio_file, sr=None)

            segment_dur_secs = 5
            segment_length = sr * segment_dur_secs

            num_sections = int(np.ceil(len(wave) / segment_length))
            print(num_sections)
            print(len(wave))
            print(segment_length)
            print(np.ceil(len(wave) / segment_length))

            split = []

            for i in range(num_sections):
                t = wave[i * segment_length: (i + 1) * segment_length]
                split.append(t)

            for i in range(num_sections):
                recording_name = os.path.basename(audio_file[:-4])
                out_file = f"{recording_name}{str(i)}.wav"
                sf.write(os.path.join(out_dir, out_file), split[i], sr)