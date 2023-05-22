# Multi-Modal_korean_emotion_recogntion

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
