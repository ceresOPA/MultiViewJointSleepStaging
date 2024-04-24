import scipy.signal as sci_signal
import numpy as np

# Butterworth低通滤波器
def butter_lowpass_filter(data,order,cutoff,fs):
    wn = 2*cutoff/fs
    b, a = sci_signal.butter(order, wn, 'lowpass', analog = False)
    output = sci_signal.filtfilt(b, a, data, axis=0)

    return output

data = np.load("../sleep-cassette/npz/SC4001.npz")
X = data['x'][0]
output = butter_lowpass_filter(X, 8, 30, 100)
print("output", output.shape)

import matplotlib.pyplot as plt

plt.figure(figsize=(16,6))
plt.plot(output[:600])
plt.savefig("signal_filter.png")