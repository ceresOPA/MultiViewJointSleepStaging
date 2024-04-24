import numpy as np
from tqdm import tqdm
import glob


# Butterworth低通滤波器
def butter_lowpass_filter(data,order,cutoff,fs):
    wn = 2*cutoff/fs
    b, a = sci_signal.butter(order, wn, 'lowpass', analog = False)
    output = sci_signal.filtfilt(b, a, data, axis=1)

    return output

# 参数设置
signal_length = 30  # 信号长度，单位：秒
sample_rate = 100  # 采样频率，单位：Hz
num_samples = int(signal_length * sample_rate)  # 总样本点数

window_size = 128  # 窗口大小
overlap = int(window_size * 0.65)  # 重叠大小
max_frequency = 32  # 频率上限

import scipy.signal as sci_signal



def get_time_freq_spectra(signal):
    # 计算STFT
    frequencies, times, stft_matrix = sci_signal.stft(
        signal, fs=sample_rate, nperseg=window_size, noverlap=overlap, nfft=window_size
    )

    # 选择需要的频率范围
    freq_mask = frequencies <= max_frequency
    frequencies = frequencies[freq_mask]
    stft_matrix = stft_matrix[freq_mask]

    # 归一化STFT
    # normalized_stft = stft_matrix / np.max(np.abs(stft_matrix))
    
    # 分解复数矩阵为实部和虚部
    real_parts = np.real(stft_matrix)
    imag_parts = np.imag(stft_matrix)

    # 归一化实部和虚部
    normalized_real_parts = real_parts / np.max(np.abs(real_parts))
    normalized_imag_parts = imag_parts / np.max(np.abs(imag_parts))


    input_data = np.stack((normalized_real_parts, normalized_imag_parts), axis=0)

    return input_data


files = glob.glob("../ISRUC-Sleep/npz/*.npz")
for f in files:
    data = np.load(f)
    X = data['x'][:, 0, ::2]
    print(X.shape)
    X = butter_lowpass_filter(X, 8, 30, 100)
    y = data['y']
    print(X.shape, y.shape)
    stft_X = []
    for x in tqdm(X):
        stft_x = get_time_freq_spectra(x)
        stft_X.append(stft_x)
    stft_X = np.asarray(stft_X)
    print(stft_X.shape)

    np.savez(f"../ISRUC-Sleep/freq/{f.split('/')[-1]}", x=stft_X, y=y)
