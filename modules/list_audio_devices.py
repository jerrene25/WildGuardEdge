import pyaudio

pa = pyaudio.PyAudio()

print(f"Default input device index: {pa.get_default_input_device_info()['index']}")
print()
print("All devices:")
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    marker = " <-- DEFAULT INPUT" if info['index'] == pa.get_default_input_device_info()['index'] else ""
    if info['maxInputChannels'] > 0:
        print(f"  [{i}] {info['name']} | input channels: {info['maxInputChannels']} | "
              f"default sample rate: {info['defaultSampleRate']}{marker}")

pa.terminate()