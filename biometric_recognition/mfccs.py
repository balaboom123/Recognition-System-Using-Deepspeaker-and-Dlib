import librosa
import numpy as np
from utils.root_path import root
"""
Usage:
librosa.feature.mfcc()
    y (required): Input audio signal. Typically, it is a one-dimensional time series.
    sr (optional): Sampling rate, which represents the sampling frequency of the audio signal (in samples per second). 
        If not specified, it defaults to 22050 Hz.
    n_mfcc (optional): The number of MFCC coefficients to extract. The default value is 13.
    hop_length (optional): Hop length between windows, in terms of the number of samples. The default is 512.
    n_fft (optional): FFT window size used for MFCC calculation, in terms of the number of samples. The default is 2048.
    htk (optional): If True, it calculates MFCC coefficients in HTK-style (commonly used in ASR applications).
        Otherwise, it uses the more common MIR-style calculation. The default is False.
    n_mels (optional): The number of Mel filters. The default is 128.
    center (optional): If True, it centers the audio signal before calculating MFCC. Otherwise, it does not center it. 
        The default is True.
    norm (optional): If True, it performs L2 normalization on each coefficient. The default is False.
    htk (optional): If True, it uses HTK's DCT type II, otherwise, it uses Scipy's DCT type II. The default is True.
"""
def extract_mfcc_features(audio_file, n_mfcc=13, n_fft=2048, hop_length=512):
    """

    :param audio_file:
    :param n_mfcc:
    :param n_fft:
    :param hop_length:
    :return:
    """
    # 加载音频文件
    signal, sr = librosa.load(audio_file, sr=None)

    # 提取 MFCC 特征
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # 将 MFCCs 转换为向量（可通过平均、最大化等方式）
    mfccs_vector = np.mean(mfccs, axis=1)

    return mfccs_vector


# 用法示例
audio_file = f"{root}\data\\voice_data\Voice_S1252001.MP3"  # 替换为音频文件的路径
mfcc_vector = extract_mfcc_features(audio_file)
print(mfcc_vector)
