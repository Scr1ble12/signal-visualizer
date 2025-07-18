#  Signal Visualizer App

This is a real-time **Signal Visualizer App** that captures microphone input or plays a `.wav` audio file and visualizes:

-  Waveform (Amplitude vs Time)
-  Frequency Spectrum (via FFT)
-  Spectrogram (Frequency over Time)

Built with **Python**, **PyQt5**, and **Matplotlib**, this app showcases real-time signal processing in a visual and interactive way.

---

##  Features

-  Real-time microphone input
-  Upload `.wav` files to analyze recorded signals
-  Spectrogram for frequency-over-time insights
-  Built-in Fast Fourier Transform (FFT) calculations

---

##  How to Run

1. **Install requirements**:
```bash
pip install numpy pyaudio matplotlib PyQt5
```
If you’re on Windows and hit issues with PyAudio:
```bash
pip install pipwin
pipwin install pyaudio
```

2. **Run the app**:
```bash
python main.py
```

---

##  Screenshots
<img width="1917" height="1016" alt="image" src="https://github.com/user-attachments/assets/ef2fe159-8903-4a27-ba75-ae99bafa47eb" />

---
---

## File Structure

- `main.py`: Main application file
- `README.md`: You're reading it!

---
