import pyaudio
import numpy as np
import time

CANDIDATES = [1, 5, 9, 19, 26]  # default + the other plausible physical mic entries
RECORD_SECONDS = 2
RATE = 44100
CHUNK = 1024

pa = pyaudio.PyAudio()

for device_index in CANDIDATES:
    try:
        info = pa.get_device_info_by_index(device_index)
        print(f"\nTesting [{device_index}] {info['name']}...")

        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK,
        )

        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        stream.stop_stream()
        stream.close()

        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio**2))
        print(f"  RMS: {rms:.5f}")

    except Exception as e:
        print(f"  FAILED: {e}")

pa.terminate()