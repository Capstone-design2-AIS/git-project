import librosa
import scipy.signal as signal
import numpy as np

x_data = []
y_data = []
file_name = ["V1.wav", "V2.wav", "V3.wav", "V4.wav", "V10.wav",]
# 도는 [1,0], 레는 [0,1]로 설정한다
lable = [[1,0],[0,1],[2,0],[0,2],[1,1]] 

for index in range(len(file_name)):
    audio_sample, sampling_rate = librosa.load(file_name[index], sr = None)
    input_data = np.abs(librosa.stft(audio_sample, n_fft=1024, hop_length=512, win_length = 1024, window=signal.hann))
    
    # input_data의 배열 형태를 알아본다
    shape = np.shape(input_data)
    nb_samples = shape[0]
    nb_windows = shape[1]
    
    # [nb_samples][nb_windows] 형태이므로, 딥러닝을 수행하기 위한 배열 형태 [nb_windows][nb_samples]로 맞춰준다
    input_data = input_data.T
    
    # 학습용 레이블 배열을 생성한다
    lable_tmp = [lable[index] for row in range(nb_windows)]
    
    # 하나의 배열로 합친다    
    if index == 0:
        x_data = input_data
        y_data = lable_tmp
    else:
        x_data = np.r_[x_data, input_data]
        y_data = np.r_[y_data, lable_tmp]

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

shape = np.shape(x_data)
nb_samples_x = shape[1]

shape = np.shape(y_data)
nb_samples_y = shape[1]

tf.reset_default_graph()

#set neural network model
input_size = nb_samples_x
lable_size = nb_samples_y
layer1_size = nb_samples_x*2
layer2_size = nb_samples_x*2

X = tf.placeholder(tf.float32, [None, input_size])
Y = tf.placeholder(tf.float32, [None, lable_size])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape=[input_size, layer1_size], initializer=tf.keras.initializers.he_normal(seed=None))
b1 = tf.Variable(tf.zeros([layer1_size]))

L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.get_variable("W2", shape=[layer1_size, layer2_size], initializer=tf.keras.initializers.he_normal(seed=None))
b2 = tf.Variable(tf.zeros([layer2_size]))

L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),b2))
L2 = tf.nn.dropout(L2, keep_prob)
    
W3 = tf.get_variable("W3", shape=[layer2_size, lable_size], initializer=tf.keras.initializers.he_normal(seed=None))
b3 = tf.Variable(tf.zeros([lable_size]))

model = tf.add(tf.matmul(L2, W3), b3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))

optimizer = tf.train.AdamOptimizer(0.0002).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(10):         
    _, cost_val = sess.run([optimizer, cost],                                   
                                   feed_dict={X: x_data,                                              
                                              Y: y_data,                                        
                                              keep_prob: 1.0})              

    print('Epoch:', '%04d' % (epoch ),'ave. cost =', '{:.10f}'.format(cost_val)) 
        
result = sess.run(model, feed_dict={X:x_data, keep_prob: 1.0})

print(result)


import operator
from matplotlib import pyplot as plt

array_len = len(result)
index = np.zeros(array_len)
value = np.zeros(array_len)
for i in range(array_len):
    index[i], value[i] = max(enumerate(result[i]), key=operator.itemgetter(1))

plt.plot(index)
plt.show()
