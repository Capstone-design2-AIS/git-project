##### 단순히 목소리 녹음을 위한 부분 #####

import pyaudio #마이크를 사용하기 위한 라이브러리
import wave #.wav 파일을 저장하기 위한 라이브러리
import os
######## 음성 데이터를 녹음 해 저장하는 부분 ########
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100 #비트레이트 설정
CHUNK = int(RATE / 10) # 버퍼 사이즈 1초당 44100비트레이트 이므로 100ms단위
RECORD_SECONDS = 5 #녹음할 시간 설정
#WAVE_OUTPUT_FILENAME = "./data/train/1/baecheolsu15.wav"
WAVE_OUTPUT_FILENAME = "./train/3/"
FILE_NAME = "훈현짜응"

files = os.listdir(WAVE_OUTPUT_FILENAME)
wave_count = 1;
     #폴더 이름과 그 폴더에 속하는 파일 갯수 출력
for wav in files: 
    if not wav.endswith(".wav"):continue
    else: wave_count = wave_count+1


WAVE_OUTPUT_FILENAME = WAVE_OUTPUT_FILENAME+FILE_NAME+str(wave_count)+".wav"
print(str(wave_count)+"개의 .wav존재!",WAVE_OUTPUT_FILENAME)
p = pyaudio.PyAudio() # 오디오 객체 생성

stream = p.open(format=FORMAT, # 16비트 포맷
                channels=CHANNELS, #  모노로 마이크 열기
                rate=RATE, #비트레이트
                input=True,
                frames_per_buffer=CHUNK) # CHUNK만큼 버퍼가 쌓인다.

print("Start to record the audio.")

frames = [] # 음성 데이터를 채우는 공간

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)): 
    #지정한  100ms를 몇번 호출할 것인지 10 * 5 = 50  100ms 버퍼 50번채움 = 5초
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording is finished.")

stream.stop_stream() # 스트림닫기
stream.close() # 스트림 종료
p.terminate() # 오디오객체 종료

wf = wave.open( WAVE_OUTPUT_FILENAME, 'wb') 
# WAVE_OUTPUT_FILENAME의 파일을 열고 데이터를 쓴다.
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
