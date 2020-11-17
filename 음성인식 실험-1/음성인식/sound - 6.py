import numpy as np
from matplotlib import pyplot as plt


# mean and standard deviation
mean, sigma = 0, 0.1 
sr = 8000
time_s = 5
sine_freq = 1000
sig_amp = 0.0001
sample = sr * time_s

#파이썬은 입력 값이 -1~1 사이로 계산되기 때문에 계수 값을 매우 작게 설정해야 한다.
#그러나, 값들을 직관적으로 보기 위해 short의 max+1 값을 곱하거나 나누어 사용하였다. 
short_max = 32768

# 5초짜리 화이트 노이즈 생성
s = sig_amp*np.random.normal(mean, sigma, sample) * short_max
plt.plot(s)
plt.show()

# 5초짜리 1000Hz 톤 신호 생성
x = np.arange(sample)
y = sig_amp*np.sin(2 * np.pi * sine_freq * x / sr) * short_max
plt.plot(x[0:100], y[0:100])
plt.show()

# 두 신호를 합성
tone_white =  y+s 

#FFT
import librosa
import scipy.signal as signal
import librosa.display
import numpy as np

#normalize_function
min_level_db = -100
def _normalize(S):
    return np.clip((S-min_level_db)/(-min_level_db), 0, 1)

amplitude = np.abs(librosa.stft(tone_white/short_max, n_fft=1024, hop_length=512,
                                win_length = 1024, window=signal.hann))
mag_db = librosa.amplitude_to_db(amplitude)
mag_n = _normalize(mag_db)
librosa.display.specshow(mag_n, y_axis='linear', x_axis='time')
plt.show()

#LMS 알고리즘 적용
lms_size = 16
step_size = 0.0025
N = len(tone_white)-lms_size+1

filter_w = np.zeros(lms_size) 
filterout_y = np.zeros(N)
err = np.zeros(N)
for n in range(N):
    x_in = tone_white[n:n+lms_size]
    filterout_y[n] = np.dot(x_in, filter_w.T)
    err[n] = tone_white[n + lms_size -1] - filterout_y[n]
    # 작거나 0인 값이 나오면 결과값이 이상하게 출력된다. 이를 방지하기 위해 10을 더해준다.
    sum_val = sum(x_in) + 10
    filter_w = filter_w + step_size * err[n] * (x_in / sum_val)

amplitude = np.abs(librosa.stft(err/short_max, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
mag_db = librosa.amplitude_to_db(amplitude)
mag_n = _normalize(mag_db)
librosa.display.specshow(mag_n, y_axis='linear', x_axis='time')
plt.show()
