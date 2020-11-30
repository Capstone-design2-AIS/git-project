#######################이건 함수 테스트 하는부분
import librosa
import pyaudio #마이크를 사용하기 위한 라이브러리
import wave
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression#텐서플로우로 바꿀예정
import os
##### 변수 설정 부분 #####
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 #비트레이트 설정
CHUNK = int(RATE / 10) # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5 #녹음할 시간 설정
WAVE_OUTPUT_FILENAME = "output.wav"
DATA_PATH = "./data/"
X_train = []#train_data 저장할 공간
X_test = []
Y_train = []
Y_test = []
def load_wave_generator(path): 
       
    batch_waves = []
    labels = []
    X_data = []
    Y_label = []
    idx = 0
    global X_train, X_test, Y_train, Y_test
    folders = os.listdir(path)

    for folder in folders:
        if not os.path.isdir(path):continue #폴더가 아니면 continue                   
        files = os.listdir(path+"/"+folder)        
        print("Foldername :",folder,"-",len(files))
        #폴더 이름과 그 폴더에 속하는 파일 갯수 출력
        for wav in files:
            if not wav.endswith(".wav"):continue
            else:               
                print("Filename :",wav)#.wav 파일이 아니면 continue
                y, sr = librosa.load(path+"/"+folder+"/"+wav)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
                X_data.extend(mfcc)
                label = [0 for i in range(len(folders))]
                label[idx] = 1
                for i in range(len(mfcc)):
                    Y_label.append(label)       
        idx = idx+1
    #end loop
    print("X_data :",np.shape(X_data))
    print("Y_label :",np.shape(Y_label))
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_label)
    
    #3d to 2d
#     nsamples, nx, ny = np.shape(X_train)
#     X_train = np.reshape(X_train,(nsamples,nx*ny))
#     nsamples, nx, ny = np.shape(X_test)
#     X_test = np.reshape(X_test,(nsamples,nx*ny))    
    
#     Y_train = np.argmax(Y_train, axis=1)###one-hot을 합침
#     Y_test = np.argmax(Y_test, axis=1)###one-hot을 합침
    xy = (X_train, X_test, Y_train, Y_test)
    np.save("./data.npy",xy)
    #print(X_data)
    #print(Y_label)
                
load_wave_generator(DATA_PATH)

#t = np.array(X_train);
#print("!!!!!!!!",t,t.shape,X_train)
print("X_train :",np.shape(X_train))
print("X_test :",np.shape(X_test))
print("Y_train :",np.shape(Y_train))
print("Y_test :",np.shape(Y_test))

clf = LogisticRegression()
clf.fit(X_train, np.argmax(Y_train, axis=1))


############### 일반 머신러닝에서 전체적인 정확도 측정 ###########

y_test_estimated = clf.predict(X_test)

# 정답률 구하기 
ac_score = metrics.accuracy_score(np.argmax(Y_test, axis=1), y_test_estimated)
print("정답률 =", ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))


#y, sr = librosa.load("./youinna16.wav")
#y, sr = librosa.load("./baecheolsu15.wav")
y, sr = librosa.load("./V1.wav")

mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr*0.01),n_fft=int(sr*0.02)).T
print(mfcc.shape)

y_test_estimated = clf.predict(mfcc)

# 정답률 구하기 
ac_score = metrics.accuracy_score(np.full(len(mfcc),0), y_test_estimated)
print("정답률 =", ac_score)
print(pd.value_counts(pd.Series(y_test_estimated)))

