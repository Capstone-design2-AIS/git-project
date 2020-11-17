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
librosa.display.specshow(mag_n, y_axis='linear', x_axis='time', sr=sampling_rate)
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
librosa.display.specshow(mag_n, y_axis='linear', x_axis='time', sr=sampling_rate)
plt.show()
