import pyttsx3
import subprocess
engine = pyttsx3.init()

""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate', 180)     # setting up new voice rate

voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female

def speak(text):
    engine.say(text)
    engine.runAndWait()
    
print("Press 'Quit' to Quit")

while True:
    text = input("Enter the text you want to convert to speech: ")
     # if keyboard.is_pressed("q"):
    if 'Quit' in text:
            print("Exit")
            break
    elif 'open test 3' in text:
        subprocess.Popen(['/Users/adityasoni234/Desktop/test 3'])
    else:
        speak(text)
        print("Press 'Quit' to quit or other any key to continue.")