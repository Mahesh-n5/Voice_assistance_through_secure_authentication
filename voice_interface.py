import os
import cv2
import face_recognition
import numpy as np
import datetime
import speech_recognition as sr
import pyttsx3
import subprocess
import json
import tempfile
import time

FACE_FOLDER = "faces"
NLP_PROCESSOR = "nlp_processor.py"

class VoiceInterface:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognized_user = None
        self.engine = pyttsx3.init()
        self.nlp_process = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 0.9)
        
    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def get_time_based_greeting(self):
        """Return appropriate greeting based on time of day"""
        hour = datetime.datetime.now().hour
        if 5 <= hour < 12:
            return "Good morning"
        elif 12 <= hour < 17:
            return "Good afternoon"
        elif 17 <= hour < 21:
            return "Good evening"
        else:
            return "Good night"
    
    def load_face_data(self):
        """Load known face encodings"""
        if not os.path.exists(FACE_FOLDER):
            os.makedirs(FACE_FOLDER)
            print(f"Created faces directory at {FACE_FOLDER}")
            return False
            
        for filename in os.listdir(FACE_FOLDER):
            if filename.startswith('.'):
                continue
                
            img_path = os.path.join(FACE_FOLDER, filename)
            try:
                image = face_recognition.load_image_file(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if max(image.shape) > 2000:
                    image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
                    
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) == 0:
                    print(f"⚠️ No face detected in {filename}")
                    continue
                    
                encoding = face_recognition.face_encodings(image, face_locations)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(os.path.splitext(filename)[0])
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
        print(f" Loaded {len(self.known_face_names)} face encodings")
        return len(self.known_face_names) > 0
        
    def initialize(self):
        """Initialize speech recognizer"""
        try:
            print("Initializing speech recognizer...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            print("Speech recognizer initialized")
            return True
        except Exception as e:
            print(f"Speech recognizer init failed: {e}")
            return False
    
    def listen_for_wake_word(self):
        """Listen for the wake word 'Hey Pixel'"""
        print("🔊 Listening for 'Hey Pixel'...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                while True:
                    audio = self.recognizer.listen(source, timeout=None, phrase_time_limit=3)
                    try:
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        if "hey pixel" in text:
                            print("\n🎤 Wake word detected! Starting face authentication...")
                            return True
                    except sr.UnknownValueError:
                        continue  # Ignore unintelligible audio
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                        self.speak("Sorry, there was an issue with speech recognition. Trying again.")
                        return False
        except KeyboardInterrupt:
            return False
        except Exception as e:
            print(f"Error in wake word detection: {e}")
            return False
    
    def authenticate_face(self):
        """Authenticate user via face recognition"""
        if len(self.known_face_encodings) == 0:
            return False
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" Could not open camera")
            return False
            
        print("📷 Camera activated - Looking for recognized face...")
        authenticated = False
        start_time = datetime.datetime.now()
        timeout = 10
        
        while (datetime.datetime.now() - start_time).seconds < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                if True in matches:
                    best_match_index = np.argmin(face_recognition.face_distance(self.known_face_encodings, face_encoding))
                    self.recognized_user = self.known_face_names[best_match_index]
                    authenticated = True
                    break
                    
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            if authenticated:
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return authenticated
    
    def start_nlp_processor(self):
        """Start the NLP processor as a subprocess"""
        if not os.path.exists(NLP_PROCESSOR):
            print(f" NLP processor script not found at {NLP_PROCESSOR}")
            return False
            
        try:
            self.nlp_process = subprocess.Popen(
                ["python", NLP_PROCESSOR],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffering
            )
            # Wait for NLP processor to signal readiness
            start_time = time.time()
            timeout = 15  # Increased timeout to 15 seconds
            while time.time() - start_time < timeout:
                if self.nlp_process.poll() is not None:
                    stderr_data = self.nlp_process.stderr.read()
                    print(f"NLP processor failed to start: {stderr_data}")
                    return False
                line = self.nlp_process.stdout.readline().strip()
                if "NLP Processor ready" in line:
                    print("NLP processor started successfully")
                    return True
                time.sleep(0.1)  # Short sleep to prevent busy-waiting
            print("NLP processor failed to start: Timeout waiting for ready signal")
            self.nlp_process.terminate()
            return False
        except Exception as e:
            print(f"Error starting NLP processor: {e}")
            return False
    
    def send_to_nlp(self, command):
        """Send command to NLP processor and get response"""
        if not self.nlp_process or self.nlp_process.poll() is not None:
            if not self.start_nlp_processor():
                return None
                
        try:
            # Send command
            self.nlp_process.stdin.write(f"{command}\n")
            self.nlp_process.stdin.flush()
            
            # Get response with a timeout
            start_time = time.time()
            while time.time() - start_time < 5:  # 5-second timeout for response
                if self.nlp_process.poll() is not None:
                    print(f"NLP processor crashed: {self.nlp_process.stderr.read()}")
                    self.nlp_process = None
                    return None
                line = self.nlp_process.stdout.readline().strip()
                if line:
                    return line if line != "Not found" else None
                time.sleep(0.1)
            print("NLP processor response timeout")
            return None
        except Exception as e:
            print(f"Error communicating with NLP processor: {e}")
            return None
    
    def listen_for_command(self, authenticated):
        """Listen for voice commands"""
        if authenticated and self.recognized_user:
            greeting = self.get_time_based_greeting()
            welcome_message = f"{greeting}, {self.recognized_user}. How can I help you?"
        else:
            welcome_message = "How can I help you?"
            
        self.speak(welcome_message)
        
        print("\nListening for your command... (Say 'exit' to stop)")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                
                if 'exit' in command:
                    self.speak("Goodbye!")
                    return False
                
                # First try NLP processor
                nlp_response = self.send_to_nlp(command)
                if nlp_response:
                    self.speak(nlp_response)
                else:
                    # Fallback to built-in commands
                    self.process_builtin_command(command)
                    
                return True
                
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything. Going back to sleep.")
                return False
            except sr.UnknownValueError:
                self.speak("I didn't understand that. Please try again.")
                return True
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                self.speak("Sorry, there was an issue with speech recognition. Please try again.")
                return True
            except Exception as e:
                print(f"Error in voice recognition: {e}")
                return False
    
    def process_builtin_command(self, command):
        """Process built-in commands"""
        if 'time' in command:
            current_time = datetime.datetime.now().strftime("%I:%M %p")
            self.speak(f"The current time is {current_time}")
        elif 'date' in command:
            current_date = datetime.datetime.now().strftime("%B %d, %Y")
            self.speak(f"Today's date is {current_date}")
        elif 'play music' in command:
            self.speak("Playing your favorite music")
        elif 'who are you' in command or 'your name' in command:
            self.speak("I'm Pixel, your personal voice assistant")
        elif 'thank you' in command:
            self.speak("You're welcome!")
        else:
            self.speak("I'm not sure how to help with that yet")
    
    def run(self):
        """Main assistant loop"""
        if not self.load_face_data():
            print(" No face data loaded - authentication disabled")
            
        if not self.initialize():
            return
            
        try:
            while True:
                if self.listen_for_wake_word():
                    authenticated = self.authenticate_face() if len(self.known_face_encodings) > 0 else False
                    print("\n Ready for commands!" if authenticated else "\n❌ Unknown user")
                    while self.listen_for_command(authenticated):
                        pass
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.nlp_process and self.nlp_process.poll() is None:
            self.nlp_process.terminate()

if __name__ == "__main__":
    assistant = VoiceInterface()
    assistant.run()