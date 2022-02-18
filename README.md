# mediapipe-sface-sample
[MediaPipe Face Detection](https://github.com/google/mediapipe)で検出した顔画像にSFaceを用いて顔認証を行うサンプルです。<br>
SFaceのONNXモデルは[opencv/opencv_zoo](https://github.com/opencv/opencv_zoo)から取得しています。<br>
![0114](https://user-images.githubusercontent.com/37477845/154732378-999a50c1-af32-481e-8cb9-bed0729b2a0d.gif)

# Requirement 
* mediapipe 0.8.8 or later
* OpenCV 3.4.2 or later
* onnxruntime 1.9.0 or later

# Demo
```bash
python sample_facedetection.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model_selection<br>
モデル選択(0：2m以内の検出に最適なモデル、1：5m以内の検出に最適なモデル)<br>
デフォルト：0
* --min_detection_confidence<br>
検出信頼値の閾値<br>
デフォルト：0.5
* --sface_model<br>
SFaceのONNXモデル格納パス<br>
デフォルト：model/face_recognition_sface_2021dec.onnx
* --sface_input_shape<br>
SFaceの入力形状<br>
デフォルト：112,112
* --sface_score_th<br>
顔認証閾値<br>
デフォルト：0.25

# Reference
* [MediaPipe](https://github.com/google/mediapipe)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
