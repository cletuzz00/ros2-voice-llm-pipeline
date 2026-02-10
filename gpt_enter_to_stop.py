import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI
import whisper

# auto pick device indices with sounddevice
# Setting to None uses system default devices
INPUT_DEVICE_INDEX = None   # System default microphone
OUTPUT_DEVICE_INDEX = None  # System default speakers

# Print what devices will be used
if INPUT_DEVICE_INDEX is None and OUTPUT_DEVICE_INDEX is None:
    default_in, default_out = sd.default.device
    print(f"[INFO] Using system default audio devices:")
    print(f"  Input: {sd.query_devices(default_in)['name']}")
    print(f"  Output: {sd.query_devices(default_out)['name']}")
elif INPUT_DEVICE_INDEX is not None and OUTPUT_DEVICE_INDEX is not None:
    print(f"[INFO] Using specified audio devices:")
    print(f"  Input: {sd.query_devices(INPUT_DEVICE_INDEX)['name']}")
    print(f"  Output: {sd.query_devices(OUTPUT_DEVICE_INDEX)['name']}")
else:
    if INPUT_DEVICE_INDEX is not None:
        print(f"[INFO] Using specified input device:")
        print(f"  Input: {sd.query_devices(INPUT_DEVICE_INDEX)['name']}")
    if OUTPUT_DEVICE_INDEX is not None:
        print(f"[INFO] Using specified output device:")
        print(f"  Output: {sd.query_devices(OUTPUT_DEVICE_INDEX)['name']}")

# AUDIO SETTINGS
SAMPLE_RATE = 16000          # Whisper-friendly
CHANNELS = 1                 # mic input channel
CHUNK_SECONDS = 0.2         # recording chunk size (stop checked after each chunk)
MIN_UTTERANCE_SECONDS = 0.5  # skip processing if recording shorter than this

WHISPER_MODEL_NAME = "base"  # Whisper base model
OPENAI_CHAT_MODEL = "gpt-4o"

TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_SR_HINT = 24000          # typical output SR for OpenAI TTS wav

# LOAD ENV + OPENAI CLIENT

env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Ensure .env is next to this script.")

openai_client = OpenAI(api_key=api_key)

# LOAD WHISPER
print("[INFO] Loading Whisper...")
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
print(f"[INFO] Whisper loaded: {WHISPER_MODEL_NAME}")

# AUDIO SETUP
sd.default.device = (INPUT_DEVICE_INDEX, OUTPUT_DEVICE_INDEX)
sd.default.samplerate = SAMPLE_RATE

# SYSTEM PROMPT (YOUR REQUEST)
SYSTEM_PROMPT = """
You are the conversational brain for a robot voice pipeline used in a Human–Robot Interaction project.

Context:
- The user speaks into a microphone.
- Whisper transcribes the speech to text.
- You (the LLM) generate a helpful, spoken response.
- A text-to-speech system will read your response aloud to the user.

Behavior rules:
- Keep responses concise (1–4 sentences unless the user asks for detail).
- Speak naturally, like a friendly robot assistant.
- If the transcription seems incomplete or unclear, ask a short follow-up question.
- Do not mention internal tool names (Whisper, Ollama, OpenAI TTS, ROS).
- If the user says "stop", "quit", or "exit", respond with a short goodbye.

Topic focus:
- You are a Data Science Master's student at the University of Michigan Tech.
"""

# Keep conversation history (last K turns)
conversation_memory = []  # list of {"role": "...", "content": "..."}
MAX_TURNS_TO_KEEP = 10

def add_to_memory(role: str, content: str):
    conversation_memory.append({"role": role, "content": content})
    if len(conversation_memory) > 2 * MAX_TURNS_TO_KEEP + 1:
        del conversation_memory[:2]

# ENTER-TO-STOP RECORDING (variable length)
def record_until_enter() -> np.ndarray:
    stop_event = threading.Event()
    chunks = []
    n_samples = int(CHUNK_SECONDS * SAMPLE_RATE)

    def recorder():
        while not stop_event.is_set():
            chunk = sd.rec(
                n_samples,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocking=True,
            )
            chunk = np.squeeze(chunk).astype(np.float32)
            chunks.append(chunk)

    thread = threading.Thread(target=recorder)
    thread.start()
    input("Press Enter when done speaking... ")
    stop_event.set()
    thread.join()

    if not chunks:
        return np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)  # 0.1s silence
    return np.concatenate(chunks)

# TRANSCRIBE (ENGLISH ONLY)
def transcribe(audio: np.ndarray) -> str:
    print(" Transcribing...")
    result = whisper_model.transcribe(
        audio,
        language="en",
        task="transcribe",
        fp16=False
    )
    text = (result.get("text") or "").strip()
    print("You said:", text)
    return text

# OPENAI CHAT COMPLETIONS
def llm_response(user_msg: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    for item in conversation_memory[-2 * MAX_TURNS_TO_KEEP:]:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": user_msg})

    response = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=messages,
    )
    reply = (response.choices[0].message.content or "").strip()
    return reply

# OPENAI TTS SPEAK
def speak_openai_tts(text: str, out_wav: str = "tts_output.wav"):
    if not text:
        return

    print("Speaking:", text)

    response = openai_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text
    )

    with open(out_wav, "wb") as f:
        f.write(response.read())

    audio, sr = sf.read(out_wav)
    sd.play(audio, sr)
    sd.wait()

# MAIN LOOP (Enter to start, Enter to stop)
def main():
    print("\nREADY — Enter to start speaking, Enter again when done (variable length).\n")

    while True:
        input("Press Enter to start speaking... ")
        audio = record_until_enter()

        duration_sec = len(audio) / SAMPLE_RATE
        if duration_sec < MIN_UTTERANCE_SECONDS:
            print("[INFO] Recording too short. Try again.")
            continue

        text = transcribe(audio)

        if not text:
            print("[INFO] Empty transcription. Skipping.")
        else:
            lower = text.lower().strip()
            if lower in ["exit", "quit", "stop"]:
                goodbye = "Shutting down. Goodbye."
                speak_openai_tts(goodbye)
                break

            add_to_memory("user", text)

            try:
                reply = llm_response(text)
            except Exception as e:
                reply = "I had trouble reaching the language model. Please check that your local LLM is running."
                print("[ERROR] LLM call failed:", e)

            if not reply:
                reply = "I'm not sure I caught that. Could you repeat it?"

            add_to_memory("assistant", reply)
            print("Assistant:", reply)
            speak_openai_tts(reply)

if __name__ == "__main__":
    main()
