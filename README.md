# face_voice_recognition
## introduction
This project is for classifying the face in a group. It can be used to manage the photo in 
the family, recognize employees in a company, etc. It will process in this procedure.
1. collect the person’s facial pictures and sort it in specific file structure.
2. the model (face_recognition in dlib) will turn every picture into an array with 128 
numbers. 
3. using SVM to classify the face
4. Input an image and clarify who it is

## turn face into 128 dimensions vector
use face_descript_compute.py to compute dacial description
```mermaid
flowchart LR
A(collect image) -->|cnn_face_detection_model| B(128D vector)
B --> C(classification)
```

## File structure  
- root
  - data/
    - person_1/
      - person_1_face-1.jpg
      - person_1_face-2.jpg
      - ...
      - person_1_face-n.jpg
    - person_2/
      - person_2_face-1.jpg
      - person_2_face-2.jpg
      - ...
      - person_2_face-n.jpg
    - ...
    - person_n/
      - person_n_face-1.jpg
      - person_n_face-2.jpg
      - ...
      - person_n_face-n.jpg
  - test/
    - file1.jpg
    - file2.jpg
---------------------------
## Issue
1. Face Recognition Model Accuracy:
The face_recognition model in dlib does not provide sufficient accuracy. In contrast, the cnn_face_detection_model_v1 in dlib, while more accurate, is less efficient.
2. Model Tuning:
There is a need to fine-tune the classification model to enhance its performance and accuracy.
3. Quantity Concern:
If the number of people is too large, the model may exhibit reduced accuracy, leading to unreliable results.
