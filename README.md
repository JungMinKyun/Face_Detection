# Face_Detection
A minimal Python wrapper around MediaPipe's Face Detection.


- Official guide: https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector/python
- This repository provides a simple reference implementation.

## 1) Requirements

``` bash
python -m pip install mediapipe opencv-python
```

## 2) Recommended project layout

Face detector can be applied for `image_array` as did in `example_code.py`
(importing is essential)

```
project_root/
├─ face_detect.py
└─ {application_code}
```

## 3) Model download
You don't need to download the `.tflite` file manually in this repository.
`face_detect.py` already includes code that downloads the model automatically.


## 4) About the Output
The simple output of `example_code.py`.
```
0-th Image                        # Sinlge face Image
{0: ((82, 93), (154, 165))}
{0: 0.8479098677635193}

1-th Image                        # Three faces Image
{0: ((195, 54), (228, 87)), 1: ((112, 40), (150, 78)), 2: ((41, 49), (82, 90))}
{0: 0.9066254496574402, 1: 0.7117299437522888, 2: 0.6115265488624573}

2-th Image                        # No face Image
{}
{}
```
In application, the input of image array must be 'RGB' format.
