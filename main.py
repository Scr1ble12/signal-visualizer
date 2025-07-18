import sys
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wave
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                             QFileDialog, QLabel, QHBoxLayout)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class AudioStream:
    def __init__(self, rate=44100, chunk=1024):
        self.rate = rate
        self.chunk = chunk
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)

    def read(self):
        data = self.stream.read(self.chunk, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class SignalVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Visualizer with File Upload & Spectrogram")
        self.setGeometry(100, 100, 900, 700)

        self.audio = AudioStream()
        self.use_file = False
        self.file_data = None
        self.file_rate = None
        self.file_index = 0
        self.chunk = 1024

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.canvas = FigureCanvas(plt.Figure())
        layout.addWidget(self.canvas)

        self.ax_waveform = self.canvas.figure.add_subplot(311)
        self.ax_fft = self.canvas.figure.add_subplot(312)
        self.ax_spec = self.canvas.figure.add_subplot(313)

        btn_layout = QHBoxLayout()

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.close)

        self.upload_button = QPushButton("Upload WAV File")
        self.upload_button.clicked.connect(self.load_file)

        btn_layout.addWidget(self.stop_button)
        btn_layout.addWidget(self.upload_button)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.timer = self.canvas.new_timer(100)
        self.timer.add_callback(self.update_plot)
        self.timer.start()

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "WAV Files (*.wav)")
        if file_path:
            wf = wave.open(file_path, 'rb')
            self.file_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            self.file_data = np.frombuffer(frames, dtype=np.int16)
            self.file_index = 0
            self.use_file = True

    def get_audio_chunk(self):
        if self.use_file and self.file_data is not None:
            end_index = self.file_index + self.chunk
            if end_index >= len(self.file_data):
                self.file_index = 0
            chunk = self.file_data[self.file_index:end_index]
            self.file_index += self.chunk
            return chunk
        else:
            return self.audio.read()

    def update_plot(self):
        data = self.get_audio_chunk()

        self.ax_waveform.clear()
        self.ax_fft.clear()
        self.ax_spec.clear()

        self.ax_waveform.plot(data)
        self.ax_waveform.set_title("Waveform")

        fft_data = np.abs(np.fft.fft(data))[:len(data)//2]
        freqs = np.fft.fftfreq(len(data), 1/(self.file_rate if self.use_file else self.audio.rate))[:len(data)//2]
        self.ax_fft.plot(freqs, fft_data)
        self.ax_fft.set_title("Frequency Spectrum")

        self.ax_spec.specgram(data, Fs=(self.file_rate if self.use_file else self.audio.rate), NFFT=256, noverlap=128)
        self.ax_spec.set_title("Spectrogram")

        self.canvas.draw()

    def closeEvent(self, event):
        self.audio.close()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignalVisualizer()
    window.show()
    sys.exit(app.exec_())