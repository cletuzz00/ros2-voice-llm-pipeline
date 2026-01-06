# HRI Voice Pipeline

A Human-Robot Interaction (HRI) voice pipeline system that enables real-time conversational interaction with robots. The system combines speech-to-text (STT), a local language model (LLM), and text-to-speech (TTS) to create a complete voice interaction loop.

## Features

- **Speech-to-Text**: Real-time audio transcription using OpenAI Whisper
- **Language Model**: Local LLM inference via Ollama (Mistral model)
- **Text-to-Speech**: High-quality voice synthesis using OpenAI TTS
- **ROS2 Integration**: Modular ROS2 nodes for distributed processing
- **Standalone Mode**: Single-script implementation for testing and development

## Architecture

The system consists of three main components:

1. **STT Node** (`stt_node.py`): Captures audio from microphone → Whisper transcription → publishes to `/speech_text`
2. **LLM Node** (`llm_node.py`): Subscribes to `/speech_text` → generates response via Ollama → publishes to `/robot_reply`
3. **TTS Node** (`tts_node.py`): Subscribes to `/robot_reply` → generates speech via OpenAI TTS → plays audio

```
Microphone → [STT Node] → /speech_text → [LLM Node] → /robot_reply → [TTS Node] → Speakers
```

## Prerequisites

### System Requirements

- **macOS** or **Linux** (Ubuntu/Debian recommended)
- Python 3.8+
- ROS2 Humble (for ROS2 nodes) or standalone mode
- Audio input/output devices

### Required Services

- **Ollama**: Local LLM server
  ```bash
  # Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh
  
  # Start Ollama
  ollama serve
  
  # Pull Mistral model
  ollama pull mistral
  ```

- **OpenAI API Key**: For TTS functionality
  - Sign up at https://platform.openai.com/
  - Create an API key
  - Store in `.env` file (see Configuration)

## Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd ros2-voice-llm-pipeline
```

### 2. Install System Dependencies

**macOS:**
```bash
brew install portaudio
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv portaudio19-dev libsndfile1 libasound2-dev alsa-utils
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y python3-pip python3-venv portaudio-devel libsndfile alsa-lib-devel alsa-utils
```

### 3. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment

Create a `.env` file in the project root:

```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

Or for ROS2 setup:
```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > ~/.env
```

### 5. (Optional) ROS2 Setup

If using ROS2 nodes, ensure ROS2 Humble is installed:

```bash
# Source ROS2
source /opt/ros/humble/setup.bash

# Build workspace (if using ROS2 workspace structure)
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Usage

### Standalone Mode

Run the complete pipeline in a single script:

```bash
python openai_implementation.py
```

**Configuration:**
- Edit device indices in the script (lines 15-16) for your audio hardware
- Adjust recording parameters (lines 18-21)
- Modify system prompt (lines 52-70)

**Find Audio Devices:**
```python
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

### ROS2 Mode

Run each component as a separate ROS2 node:

**Terminal 1 - Speech-to-Text:**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash  # If using workspace
ros2 run hri_voice_pipeline whisper_stt \
  --ros-args \
  -p record_seconds:=4.0 \
  -p cycle_seconds:=5.0 \
  -p device_index:=-1 \
  -p model_name:=base
```

**Terminal 2 - Language Model:**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run hri_voice_pipeline llm_node \
  --ros-args \
  -p model:=mistral \
  -p ollama_url:=http://localhost:11434/api/generate \
  -p max_turns:=8
```

**Terminal 3 - Text-to-Speech:**
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 run hri_voice_pipeline openai_tts \
  --ros-args \
  -p env_path:=$HOME/.env \
  -p voice:=alloy \
  -p model:=gpt-4o-mini-tts
```

### Monitor ROS2 Topics

```bash
# View transcribed speech
ros2 topic echo /speech_text

# View LLM responses
ros2 topic echo /robot_reply
```

## Configuration

### Audio Devices

**macOS:**
```python
INPUT_DEVICE_INDEX = 0   # MacBook Air Microphone
OUTPUT_DEVICE_INDEX = 1  # MacBook Air Speakers
```

**Linux:**
- Use device indices from `sd.query_devices()` output
- Or use `-1` for system default
- For PulseAudio: `sd.default.device = ('pulse', 'pulse')`

### Model Parameters

**Whisper Models:**
- `tiny`, `base`, `small`, `medium`, `large`
- Larger models = better accuracy but slower

**Ollama Models:**
- `mistral` (default, ~4.1GB)
- `llama2`, `codellama`, `phi`, etc.
- See: `ollama list` for available models

**OpenAI TTS:**
- Model: `gpt-4o-mini-tts` (default)
- Voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

### ROS2 Parameters

**STT Node:**
- `device_index`: Audio input device (-1 = default)
- `sample_rate`: Audio sample rate (16000 recommended)
- `record_seconds`: Recording duration per cycle
- `cycle_seconds`: Time between recordings
- `model_name`: Whisper model size

**LLM Node:**
- `model`: Ollama model name
- `ollama_url`: Ollama API endpoint
- `max_turns`: Conversation history length

**TTS Node:**
- `env_path`: Path to `.env` file
- `voice`: OpenAI TTS voice name
- `model`: OpenAI TTS model
- `output_device`: Audio output device index

## Project Structure

```
HRI/
├── llm_node.py              # ROS2 LLM node
├── stt_node.py              # ROS2 Whisper STT node
├── tts_node.py              # ROS2 OpenAI TTS node
├── openai_implementation.py # Standalone complete pipeline
├── stream_test.py           # Test script with VAD
├── piper_test.py            # Alternative TTS test
├── openai_tts_test.py       # TTS testing script
├── stt.py                   # Alternative STT implementation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Troubleshooting

### Audio Issues

**No audio input detected:**
```bash
# List available devices
python3 -c "import sounddevice as sd; print(sd.query_devices())"

# Test microphone (Linux)
arecord -d 3 test.wav
aplay test.wav
```

**Permission errors (Linux):**
```bash
sudo usermod -a -G audio $USER
# Log out and back in
```

### Ollama Connection Issues

**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

**Verify model is available:**
```bash
ollama list
```

**Restart Ollama:**
```bash
# Stop service
brew services stop ollama  # macOS
systemctl --user stop ollama  # Linux

# Start manually
ollama serve
```

### OpenAI API Issues

**Verify API key:**
```bash
# Check .env file exists and contains key
cat .env
```

**Test API connection:**
```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Test call...
```

### ROS2 Issues

**Nodes not found:**
- Ensure workspace is built: `colcon build --symlink-install`
- Source setup files: `source install/setup.bash`
- Check `setup.py` entry points are correct

**Topics not publishing:**
- Verify all nodes are running
- Check topic names match: `ros2 topic list`
- Monitor with `ros2 topic echo <topic_name>`

## Development

### Testing Individual Components

**Test STT only:**
```python
import whisper
import sounddevice as sd
import numpy as np

model = whisper.load_model("base")
audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, blocking=True)
result = model.transcribe(np.squeeze(audio))
print(result["text"])
```

**Test LLM only:**
```python
import requests

payload = {
    "model": "mistral",
    "prompt": "Hello, how are you?",
    "stream": False
}
r = requests.post("http://localhost:11434/api/generate", json=payload)
print(r.json()["response"])
```

**Test TTS only:**
```python
from openai import OpenAI
import sounddevice as sd
import soundfile as sf

client = OpenAI(api_key="your-key")
response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="alloy",
    input="Hello, this is a test."
)
# Save and play audio...
```

## Performance Notes

- **Whisper**: First transcription is slower (model loading). Subsequent calls are faster.
- **Ollama**: First request loads model into RAM (~4GB for Mistral). Keep Ollama running for best performance.
- **OpenAI TTS**: Network latency depends on connection. Consider caching for repeated phrases.

## License

[Specify your license here]

## Contributing

[Contributing guidelines if applicable]

## Acknowledgments

- OpenAI Whisper for speech recognition
- Ollama for local LLM inference
- OpenAI for text-to-speech synthesis
- ROS2 community for robotics framework
