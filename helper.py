import sounddevice as sd

print("Available audio devices:")
print("=" * 80)
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"Index {i}: {device['name']} (Input)")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample rate: {device['default_samplerate']}")
        print()