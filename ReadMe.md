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

If using ROS2 nodes, ensure ROS2 Humble is installed and sourced:

```bash
# Source ROS2 Humble
source /opt/ros/humble/setup.bash
```

**Note:** The ROS2 nodes use standard `rclpy` APIs compatible with ROS2 Humble. They can be run directly with Python without requiring a ROS2 workspace or package structure.

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

Run each component as a separate ROS2 node. **Note:** These nodes use standard ROS2 Humble APIs and can be run directly with Python.

**Terminal 1 - Speech-to-Text:**
```bash
source /opt/ros/humble/setup.bash
python tts_node.py \
  --ros-args \
  -p record_seconds:=4.0 \
  -p cycle_seconds:=5.0 \
  -p device_index:=-1 \
  -p model_name:=base
```

**Using a recorded audio file instead of microphone:**
```bash
source /opt/ros/humble/setup.bash
python tts_node.py \
  --ros-args \
  -p use_audio_file:=true \
  -p audio_file_path:=hellothere.mp3 \
  -p cycle_seconds:=5.0 \
  -p model_name:=base
```

**Terminal 2 - Language Model:**
```bash
source /opt/ros/humble/setup.bash
python llm_node.py \
  --ros-args \
  -p model:=mistral \
  -p ollama_url:=http://localhost:11434/api/generate \
  -p max_turns:=8
```

**Terminal 3 - Text-to-Speech:**
```bash
source /opt/ros/humble/setup.bash
python stt_node.py \
  --ros-args \
  -p env_path:=.env \
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

### ROS2 Package Mode (Alternative)

For a proper ROS2 package structure that allows using `ros2 run` commands, use the package in the `ros2_package/` directory.

**Build the ROS2 Package:**

```bash
# Navigate to the package directory
cd ros2_package/voice_llm_pipeline

# Build the package
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

**Run nodes using ros2 run:**

**Terminal 1 - Speech-to-Text:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash  # From the package directory
ros2 run voice_llm_pipeline stt_node \
  --ros-args \
  -p record_seconds:=4.0 \
  -p cycle_seconds:=5.0 \
  -p device_index:=-1 \
  -p model_name:=base
```

**Using a recorded audio file:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run voice_llm_pipeline stt_node \
  --ros-args \
  -p use_audio_file:=true \
  -p audio_file_path:=/path/to/your/audio.wav \
  -p cycle_seconds:=5.0 \
  -p model_name:=base
```

**Terminal 2 - Language Model:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run voice_llm_pipeline llm_node \
  --ros-args \
  -p model:=mistral \
  -p ollama_url:=http://localhost:11434/api/generate \
  -p max_turns:=8
```

**Terminal 3 - Text-to-Speech:**
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run voice_llm_pipeline tts_node \
  --ros-args \
  -p env_path:=$HOME/.env \
  -p voice:=alloy \
  -p model:=gpt-4o-mini-tts
```

**Note:** The ROS2 package structure is located in `ros2_package/voice_llm_pipeline/`. This provides a proper ROS2 package that can be integrated into ROS2 workspaces. The direct Python execution method (above) is simpler and doesn't require building a package.

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

**STT Node (`tts_node.py`):**
- `device_index`: Audio input device (-1 = default)
- `sample_rate`: Audio sample rate (16000 recommended)
- `record_seconds`: Recording duration per cycle
- `cycle_seconds`: Time between recordings
- `model_name`: Whisper model size
- `use_audio_file`: Boolean to use audio file instead of microphone (default: false)
- `audio_file_path`: Path to audio file when `use_audio_file` is true

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
ros2-voice-llm-pipeline/
├── llm_node.py              # ROS2 LLM node (direct execution)
├── stt_node.py              # ROS2 OpenAI TTS node (direct execution)
├── tts_node.py              # ROS2 Whisper STT node (direct execution)
├── openai_implementation.py # Standalone complete pipeline
├── stream_test.py           # Test script with VAD
├── piper_test.py            # Alternative TTS test
├── openai_tts_test.py       # TTS testing script
├── stt.py                   # Alternative STT implementation
├── requirements.txt         # Python dependencies
├── ros2_package/            # ROS2 package structure
│   └── voice_llm_pipeline/
│       ├── package.xml       # ROS2 package manifest
│       ├── setup.py          # Python package setup
│       ├── setup.cfg         # Setup configuration
│       └── voice_llm_pipeline/
│           ├── __init__.py
│           ├── stt_node.py  # Whisper STT node
│           ├── llm_node.py   # LLM node
│           └── tts_node.py   # OpenAI TTS node
└── README.md                # This file
```

**Note:** The root-level node files (`stt_node.py`, `llm_node.py`, `tts_node.py`) can be run directly with Python. The `ros2_package/` directory contains a proper ROS2 package structure for use with `ros2 run` commands.

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

**Nodes not starting:**
- Ensure ROS2 Humble is sourced: `source /opt/ros/humble/setup.bash`
- Verify Python dependencies are installed: `pip install -r requirements.txt`
- Check that all required services (Ollama, OpenAI API) are configured

**Topics not publishing:**
- Verify all nodes are running in separate terminals
- Check topic names match: `ros2 topic list`
- Monitor with `ros2 topic echo <topic_name>`
- Ensure nodes are in the same ROS2 domain (default domain is fine)

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

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Ollama for local LLM inference
- OpenAI for text-to-speech synthesis
- ROS2 community for robotics framework
