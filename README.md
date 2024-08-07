# Recognition_System_Using_Deepspeaker_and_Dlib
## Introduction
This project is for classifying the biometric feature in a group. It can control the permission of entrance, which is more secure than the key.
For instance, recognize family member to entry home or employees in a company, etc. It will process in this procedure.

### face
1. collect the person’s facial pictures and sort it in specific file structure.  
2. detecte the boxes of face  
3. dlib shape_predictor_68_face_landmarks will get the facial feature  
4. compute features in training data to an 128D vector  
5. output to feature fusion  
  
### voice
use deep-speaker cnn model to get the feature of voice
pass  

## turn face into 128 dimensions vector
### dlib cnn face detection model
use face_descript_compute.py to compute facial description
```mermaid
flowchart LR
A(collect image) -->|cnn_face_detection| B(128D vector)
B --> C(feature fusion)
```

## File structure  
```
project
│
└───data
│   │
│   └───person_1
│       │   person_1_face-1.jpg
│       │   person_1_face-2.jpg
│       │   ...
│       │   person_1_face-n.jpg
│   │  
│   └───person_2
│       │   person_2_face-1.jpg
│       │   person_2_face-2.jpg
│       │   ...
│       │   person_2_face-n.jpg
│   ...
│   ...
│   │
│   └───person_n
│       │   person_n_face-1.jpg
│       │   person_n_face-2.jpg
│       │   ...
│       │   person_n_face-n.jpg
│   
└───test
    │   file1.jpg
    │   file2.jpg
```
