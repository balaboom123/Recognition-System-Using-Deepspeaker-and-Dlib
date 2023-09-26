# face_voice_recognition
##introduction
This project is for classifying the face in a group. It can be used to manage the photo in 
the family, recognize employees in a company, etc. It will process in this procedure.
1. collect the person’s facial pictures and sort it in specific file structure. 
2. the model (face_recognition in dlib) will turn every picture into an array with 128 
numbers. 
3. using SVM to classify the face
4. Input an image and clarify who it is

##File structure
data/
 person_1/
 person_1_face-1.jpg
 person_1_face-2.jpg
 …
 …
 person_1_face-n.jpg
 person_2/
 person_2_face-1.jpg
 person_2_face-2.jpg
 …
 …
 person_2_face-n.jpg
 …
 …
 person_n/
 person_n_face-1.jpg
 person_n_face-2.jpg
 …
 …
 person_n_face-n.jpg

 ## Issue
1. Face recognition model in dlib is not accurate enough. In contrast, 
cnn_face_detection_model_v1 in dlib is less efficient
2. Need to tuning the classification model
3. If the quantity of the people are too much, it may be less accuracy
