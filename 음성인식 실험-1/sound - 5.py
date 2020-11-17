import librosa
from python_speech_features import mfcc
import numpy as np
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


mfcc_size = 20
x_data = []
y_data = []

def compute_mfcc(audio_data, sample_rate):
    mfcc_feat = mfcc(audio_data, sample_rate, winlen=0.020, winstep=0.01,
                     numcep=mfcc_size, nfilt=26, nfft=512, lowfreq=0, highfreq=None,
                     preemph=0.97, ceplifter=22, appendEnergy=True)
    return mfcc_feat 

audio_sample, sampling_rate = librosa.load("V1.wav", sr = None)
mfcc_w = compute_mfcc(audio_sample, sampling_rate)

#woman train data
shape = np.shape(mfcc_w)
lable_tmp = [[1,0]for row in range(shape[0])]

x_data = mfcc_w
y_data = lable_tmp

#man train data
audio_sample, sampling_rate = librosa.load("V2.wav", sr = None)
mfcc_m = compute_mfcc(audio_sample, sampling_rate)

shape = np.shape(mfcc_m)
lable_tmp = [[0,1]for row in range(shape[0])]

x_data = np.r_[x_data, mfcc_m]
y_data = np.r_[y_data, lable_tmp]

print(len(x_data), len(y_data))



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

optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(20000):         
    _, cost_val = sess.run([optimizer, cost],                                   
                                   feed_dict={X: x_data,                                              
                                              Y: y_data,                                        
                                              keep_prob: 0.9})              

    if epoch%1000 == 0:
        print('횟수:', '%04d' % (epoch ),'평균 =', '{:.10f}'.format(cost_val)) 

    
#test 용 데이터 확인해보기
audio_sample, sampling_rate = librosa.load("V3.wav", sr = None)
mfcc_test_w = compute_mfcc(audio_sample, sampling_rate)
shape = np.shape(mfcc_test_w)
test_label_w_tmp = [[1,0]for row in range(shape[0])]

audio_sample, sampling_rate = librosa.load("V4.wav", sr = None)
mfcc_test_m = compute_mfcc(audio_sample, sampling_rate)
shape = np.shape(mfcc_test_m)
test_label_m_tmp = [[0,1]for row in range(shape[0])]

x_test = np.r_[mfcc_test_w, mfcc_test_m]
result = sess.run(model, feed_dict={X:x_test, keep_prob: 1.0})

test_label = np.r_[test_label_w_tmp, test_label_m_tmp]
        
import operator
from matplotlib import pyplot as plt

array_len = len(result)
index = np.zeros(array_len)
value = np.zeros(array_len)
for i in range(array_len):
    index[i], value[i] = max(enumerate(result[i]), key=operator.itemgetter(1))

correct_cnt = 0
for i in range(len(index)):
    if test_label[i][int(index[i])] == 1:
        correct_cnt += 1
print("accuracy : ", correct_cnt/len(result))
    
plt.plot(index)
plt.show()

