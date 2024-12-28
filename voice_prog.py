import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tkinter import Tk, Label, Button, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sounddevice as sd
import wave
from scipy.signal import butter, lfilter

# Path to dataset directory
auto_directory = "/Users/adityasoni234/Downloads/archive-2/audio"

# Updated audio recording settings
SAMPLE_RATE = 44100  # Hz
DURATION = 5  # seconds
RECORDING_FILENAME = "recorded_audio.wav"
CHANNELS = 1  # Mono recording
CHUNK_SIZE = 2048  # Increased buffer size

# Updated noise reduction parameters
NOISE_REDUCTION_STRENGTH = 0.2  # Increased from 0.1
LOW_CUT = 300  # Hz - Increased from 100
HIGH_CUT = 8000  # Hz - Decreased from 10000

def test_microphone():
    """Test microphone input and list available devices"""
    try:
        devices = sd.query_devices()
        print("\nAvailable audio devices:")
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}")
        
        messagebox.showinfo("Test", "Speaking a test phrase...")
        test_data = sd.rec(int(SAMPLE_RATE * 2), channels=CHANNELS)
        sd.wait()
        print("Recording level:", np.max(np.abs(test_data)))
    except Exception as e:
        print(f"Error testing microphone: {e}")

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a bandpass filter to remove unwanted frequencies"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply the bandpass filter to the audio data"""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def reduce_noise(audio_data, noise_reduction_strength=NOISE_REDUCTION_STRENGTH):
    """Apply enhanced noise reduction"""
    # Calculate noise floor from first 0.1 seconds
    noise_floor = np.mean(np.abs(audio_data[:int(SAMPLE_RATE * 0.1)]))
    # Apply stronger noise gate
    audio_data[np.abs(audio_data) < noise_floor * (1 + noise_reduction_strength)] = 0
    return audio_data

def extract_audio_features(file_path):
    """Extract audio features from the given file"""
    y, sr = librosa.load(file_path, sr=None)
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfccs, axis=1))

    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend(np.mean(spectral_centroids, axis=1))

    # Root Mean Square Energy
    rms = librosa.feature.rms(y=y)
    features.extend(np.mean(rms, axis=1))

    return np.array(features)

def readAudios(directory):
    """Read all audio files from the directory and extract features"""
    features_list = []
    labels = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                features = extract_audio_features(file_path)
                features_list.append(features)
                labels.append(root.split("/")[-1])
    return np.array(features_list), np.array(labels)

def record_audio():
    """Record audio with enhanced noise reduction"""
    try:
        # Configure sounddevice settings with improved parameters
        sd.default.samplerate = SAMPLE_RATE
        sd.default.channels = CHANNELS
        sd.default.dtype = 'float32'
        
        # Get list of audio devices and try to select a good input device
        devices = sd.query_devices()
        input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        if input_devices:
            sd.default.device = input_devices[0]  # Use the first available input device
        
        # Add countdown
        for i in range(3, 0, -1):
            messagebox.showinfo("Preparing to Record", f"Recording will start in {i} seconds...")
        
        messagebox.showinfo("Recording", "Recording now! Please speak...")

        # Record audio with increased buffer size and input gain
        audio_data = sd.rec(int(SAMPLE_RATE * DURATION), blocking=True)
        
        # Enhanced noise reduction pipeline
        audio_data = audio_data.flatten()  # Convert to 1D array
        
        # Apply stronger noise reduction
        audio_data = apply_bandpass_filter(audio_data, LOW_CUT, HIGH_CUT, SAMPLE_RATE)
        
        # Apply stronger noise gate
        noise_floor = np.mean(np.abs(audio_data[:int(SAMPLE_RATE * 0.1)]))
        audio_data[np.abs(audio_data) < noise_floor * 2] = 0  # Stronger noise gate
        
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Apply volume boost
        audio_data = audio_data * 2  # Increase volume
        
        # Normalize audio while preventing clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9  # Leave some headroom
        
        # Scale to 16-bit PCM
        scaled_audio = (audio_data * 32767).astype(np.int16)

        # Save with proper WAV parameters
        with wave.open(RECORDING_FILENAME, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(scaled_audio.tobytes())

        # Verify the recording
        if os.path.getsize(RECORDING_FILENAME) > 1000:  # Check if file is not empty
            messagebox.showinfo("Success", "Audio recorded successfully!")
            predict_audio(RECORDING_FILENAME)
        else:
            messagebox.showerror("Error", "Recording appears to be empty. Please try again.")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to record audio: {e}")

def predict_audio(file_path):
    """Predict the class of the recorded audio"""
    try:
        # Extract features and make prediction
        audio_features = extract_audio_features(file_path)
        prediction = model.predict([audio_features])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display the prediction
        messagebox.showinfo("Prediction", f"Predicted Label: {predicted_label}")

        # Plot waveform and MFCCs
        plot_audio_features(file_path)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def plot_audio_features(file_path):
    """Plot the waveform and MFCCs of the audio"""
    # Load audio for visualization
    y, sr = librosa.load(file_path, sr=None)

    # Create a new window for plots
    plot_window = Tk()
    plot_window.title("Audio Features Visualization")

    # Plot waveform
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    librosa.display.waveshow(y, sr=sr, ax=axs[0])
    axs[0].set_title("Waveform")

    # Plot MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=axs[1])
    axs[1].set_title("MFCCs")
    fig.colorbar(img, ax=axs[1])

    # Display the plots in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack()
    canvas.draw()

    plot_window.mainloop()

if __name__ == "__main__":
    # Data preparation and model training
    print("Loading and preparing data...")
    features_list, labels = readAudios(auto_directory)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        features_list, labels, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Main application window
    app = Tk()
    app.title("Audio Classifier")
    app.geometry("400x300")  # Increased height for new button

    # GUI Elements
    Label(app, text="Audio Classification", font=("Arial", 16)).pack(pady=10)
    Button(app, text="Test Microphone", command=test_microphone, width=20).pack(pady=10)
    Button(app, text="Record Audio", command=record_audio, width=20).pack(pady=10)
    Label(app, text=f"Model Accuracy: {accuracy:.2f}", font=("Arial", 12)).pack(pady=10)

    app.mainloop()