## Voice assistance through secure Authentication
My project is an assistance through secure face authentication, if the person is authenticated then the voice assistance is going to interact with the person by taking the different queries from the user and responding to then using Natural language processing, where i have already trained my NLP model with large corpus so that my model will answer the user queries even in the offline 
Dataset Structure:
Here i have used two type of datasets, as follows
1) Dataset which contains the faces of the autherized persons.
2) Dataset for query answering , it is like an questiona and answer dataset

work flow:
First and  foremost the user need to call the assistance by wake word, so that the assistance will wakeup and check the person is authorized or not by using face recognization, once if the person is authorized then it will able to ask the queries, once the use says the query then it will process the query using NLP tools like spacy, and able to answer the user query from the .pkl files 

How to use :
create an folder as "Faces" and try to add your faces and store the facial embeddings,
create an data.json which your own questions and answers and train the nlp model and save the files as answer.pkl, question_embeddings.pkl, therfore we can access these file during answering to the user queries fastly

how to run:
1) install all the packages needed like speech_recognization, cv2, numpuy, spacy etc.
2) add the faces to the "faces" folder.
3) run the command "python voice_interface.py" in the terminal.

## Implementation through hardware
components needed:
1) raspberry pi- main processor(3B+,4series is better)
2) Usb mic for wake word detection as well as taking the commands from the user.
3) small speaker for output through voice.
4) CAM-Module(it may be ESP32-CAM Module, web cam with usb or official cam module for raspberry pi).
5) Servo motor in order to check the surroundings from where the voice is recognized- it used for mic as well as for cam module
6) Sound detecting sensor, to know from which direction the sound cam so that we will turn our cam and mic in that direction.
