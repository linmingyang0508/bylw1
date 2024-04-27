import os
import librosa
import numpy as np

def analyze_audio_properties_extended(audio_dir):
    """
    遍历指定目录下的所有音频文件，并打印出它们的基本属性以及动态范围和背景噪音水平。
    """
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a')):
                file_path = os.path.join(root, file)
                y, sr = librosa.load(file_path, sr=None)
                mono_or_stereo = "Mono" if y.ndim == 1 else "Stereo"
                length_seconds = librosa.get_duration(y=y, sr=sr)
                dynamic_range_db = 20 * np.log10(np.max(np.abs(y)) / np.max(np.abs(y[y != 0])))

                # 使用librosa的功率谱密度估计计算背景噪音水平
                S, phase = librosa.magphase(librosa.stft(y))
                rms = librosa.feature.rms(S=S).mean()
                noise_level_db = 20 * np.log10(rms)

                print(f"File: {file_path}")
                print(f"Channels: {mono_or_stereo}")
                print(f"Length: {length_seconds:.2f} seconds")
                print(f"Sample Rate: {sr} Hz")
                print(f"Dynamic Range (dB): {dynamic_range_db:.2f} dB")
                print(f"Background Noise Level (dB): {noise_level_db:.2f} dB")
                print("-" * 30)

audio_dir = '../data/audio'
analyze_audio_properties_extended(audio_dir)
