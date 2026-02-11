import sounddevice as sd
import numpy as np


print("Available audio devices:")
print("=" * 80)
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"Index {i}: {device['name']} (Input)")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample rate: {device['default_samplerate']}")
        print()


print("Available audio devices:")
print("=" * 80)
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"Index {i}: {device['name']} (Input)")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample rate: {device['default_samplerate']}")
        print()

print("\n" + "=" * 80)
print("Testing audio input devices...")
print("=" * 80)

# Test specific devices
test_indices = [8, 9, 11, 12]
duration = 2  # seconds
sample_rate = 44100

for idx in test_indices:
    device_name = devices[idx]['name']
    print(f"\nRecording from Index {idx} ({device_name}) for {duration} seconds...")
    print("Speak into your headphones/microphone now!")
    
    try:
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                      channels=1, device=idx, dtype='float32')
        sd.wait()
        
        # Check if audio was captured (RMS level)
        rms = np.sqrt(np.mean(audio**2))
        print(f"  Audio level (RMS): {rms:.6f}")
        
        if rms > 0.01:
            print(f"  ✓ Audio detected - headphones may be connected here")
        else:
            print(f"  ✗ No audio detected")
    except Exception as e:
        print(f"  Error: {e}")


import sounddevice as sd
import numpy as np

print("Available audio devices:")
print("=" * 80)
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"Index {i}: {device['name']} (Input)")
        print(f"  Channels: {device['max_input_channels']}")
        print(f"  Sample rate: {device['default_samplerate']}")
        print()

print("\n" + "=" * 80)
print("Testing audio output devices...")
print("=" * 80)

# Test specific devices
test_indices = [8, 9, 11, 12]
duration = 2  # seconds
sample_rate = 44100
frequency = 440  # Hz (A4 note)

for idx in test_indices:
    device_name = devices[idx]['name']
    print(f"\nPlaying test tone on Index {idx} ({device_name}) for {duration} seconds...")
    
    try:
        # Generate a sine wave test tone
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype('float32')
        
        sd.play(audio, samplerate=sample_rate, device=idx)
        sd.wait()
        
        print(f"  ✓ Tone played successfully")
        # add a short pause between tests
        sd.sleep(1500)
    except Exception as e:
        print(f"  ✗ Error: {e}")