# Multi-Modal_Korean_Emotion_Recogntion
[pdf](/pics/paper.pdf)
## 1. 데이터셋
- [ETRI 감정 데이텃셋](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)



### en_train.json example
```
[
  {
    "wav": "Sess01_script01_M001.wav",
    "text": "어 저 지그 지금 사람 친 거야? 지금 사람 친 거 맞지? 그치?",
    "year": 19,
    "en_text": "Oh, did Zigg make friends with someone now? He made friends with someone, right? Right?"
  },
  ...
  {
    "wav": "Sess01_script01_M002.wav",
    "text": "아이 씨 그러니까 나 말렸어야지. 술 먹어서 운전 안 한다고 했잖아. b/",
    "year": 19,
    "en_text": "\"I should have stopped him. He said he wouldn't drive after drinking alcohol.\""
  }
 ]

```
-> en_text is augmented by LLMs.

### labels.csv example
```
filename,dlabel,valence,Arousal
Sess01_script01_M001,surprise,1.7,4.0
Sess01_script01_M002,angry,1.3,4.3
...

```

## 2. k-wav2vec, speechT5
![image](/pics/res.png)


## 3. 멀티모달 감정인식 프로세스
#### Translator-based data conversion
![image](/pics/data_convergion.png)
#### Emotion Recognition with Meta Attention
![image](/pics/architecture.png)


## 4. AI 기반 신호 및 텍스트를 이용한 감정인식 모델
SpeechT5에 기반한 공유된 아키텍쳐들을 이용하여 한국어 상황에서 어떻게 사전학습 모델을 이용할 수 있을 지에 해답을 제시한 프로그램
단순한 텍스트만 이용하지 않고, 시그널 정보를 함께 고려하여 사용자의 감정을 판별하는 모델임

## 5. 코드 활용
- Setup conda environment
```
conda env update -n base --file environment.yaml
```

```
# train sample code
bash run.sh
# test sample code
## setup the model weight file in infer.py
python multi_infer.py
```
