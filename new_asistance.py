import speech_recognition as sr
import pyttsx3
import os
import webbrowser

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            return command
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand that. Please try again.")
        except sr.RequestError:
            speak("There seems to be an issue with the speech recognition service.")
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Try speaking again.")
    return None

def get_input():
    command = input("Type your command: ").lower()
    return command

def open_website_or_app(command):
    websites = {
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "facebook": "https://www.facebook.com",
        "github": "https://www.github.com",
        "amazon": "https://www.amazon.com"
    }
    
    apps = {
        "safari": "open -a Safari",
        "chrome": "open -a Google\\ Chrome",
        "finder": "open -a Finder",
        "notes": "open -a Notes",
        "terminal": "open -a Terminal"
    }
    
    for site in websites:
        if site in command:
            speak(f"Opening {site}")
            webbrowser.open(websites[site])
            return
    
    for app in apps:
        if app in command:
            speak(f"Opening {app}")
            os.system(apps[app])
            return
    
    speak("Sorry, I couldn't find that website or application.")

def main():
    speak("Hello, I am your voice assistant. How can I help you today?")
    while True:
        print("Say something or type your command:")
        command = recognize_speech()
        if not command:
            command = get_input()
        
        if command:
            if "hello" in command or "hey" in command:
                speak("Hello! How can I assist you?")
            elif "how r u" in command:
                speak("I am just a program, but I'm doing great! How about you?")
            elif "who are you" in command:
                speak("I am your personal voice assistant, here to help you.")
            elif "open" in command:
                open_website_or_app(command)
            elif "exit" in command or "quit" in command:
                speak("Goodbye! Have a great day!")
                break
            else:
                speak("I am not sure how to respond to that.")

if __name__ == "__main__":
    main()