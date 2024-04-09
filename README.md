# tensorflow easy image classification

Classify the hand signs on PC camera images using Keras and TensorFlow.

contains

* Script to collect dataset from camera
* Script to building and training the model

only for native Windows local
TensorFlow 2.9

## dependencies

```bash
pip install -r requirements.txt
```

## collect.py

Running this script will start capturing the camera.

Pressing a key drawn on the screen will save the current image under `./dataset/` with the corresponding label.

When you exit with ESC, the number of saved images will be output.

![2024-04-09_21h27_00.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/488658/6b2f2611-9442-1a8f-aaed-f8ea7332b4e7.jpeg)

## training.py

Running this script will build, train, and save the model.

When the number of files is 1449, it takes about 1 minute on my machine.
(`Intel Core i7-10700`, `NVIDIA GeForce RTX 3070`)

If the camera is fixed and you are testing by yourself, the evaluate results are perfect.

## predict.py

This script infers the hand sign in the frame drawn on the screen using the SavedModel created by training.py and draws a label.

![2024-04-09_21h51_25.jpg](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/488658/08991512-8c2e-53a7-3090-cb66c3f238ac.jpeg)

## TODO

FIX

* ハイバーパラメータのチューニング
* 推論前の画処理（ノイズや無駄な部分の除去、マスク）
* モデルの再設計

ADD

* 画像全体から手を検出して、そこ切り取って推論（オブジェクト検出）
* ユーザの意思があるときにのみの機能（サインが静止したときにトリガー）
* サインではなく時系列を見たジェスチャー
