import torch
import whisper
import sounddevice as sd
import numpy as np
import webrtcvad
import requests
import time


# 1. Load Whisper STT

print("Loading Whisper...")
whisper_model = whisper.load_model("base")



# 2. Load Silero TTS

print("Loading Silero...")
tts_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_tts",
    language="en",
    speaker="v3_en"
)

SAMPLE_RATE = 48000
DEFAULT_SPEAKER = "en_90"
# list of available speakers:
# print(tts_model.speakers)



# 3. Local LLM (Mistral via Ollama) + Conversation Memory

conversation_memory = []  # Stores history

def llm_response(user_msg, model="mistral"):
    global conversation_memory

    conversation_memory.append(f"User: {user_msg}")

    prompt = "\n".join(conversation_memory[-6:]) + "\nAssistant:"

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    r = requests.post(url, json=payload)
    reply = r.json()["response"].strip()

    conversation_memory.append(f"Assistant: {reply}")

    return reply



# 4. Voice Activity Detection (VAD)

vad = webrtcvad.Vad()
vad.set_mode(2)  # 0–3 (3 = most sensitive)

FRAME_DURATION = 30  # ms
SAMPLE_RATE_VAD = 16000
FRAME_SIZE = int(SAMPLE_RATE_VAD * FRAME_DURATION / 1000)

def frame_generator(audio_bytes):
    n = FRAME_SIZE * 2
    for idx in range(0, len(audio_bytes), n):
        yield audio_bytes[idx:idx+n]

def vad_detect(audio_float32):
    audio_int16 = (audio_float32 * 32768).astype(np.int16).tobytes()
    frames = list(frame_generator(audio_int16))
    speech_frames = [f for f in frames if len(f) == FRAME_SIZE*2 and vad.is_speech(f, SAMPLE_RATE_VAD)]

    return len(speech_frames) > 0



# 5. Continuous Listening with VAD


# 5. Continuous Listening with VAD (FIXED & RELIABLE)


# Force correct mic + speaker 
sd.default.device = (0, 1)
sd.default.samplerate = SAMPLE_RATE_VAD

def listen_until_silence(threshold_silence=1.2):
    print("\n🎤 Ready. Start speaking...")

    buffer = []
    silence_start = None

    while True:
        # RECORD 100ms of audio (mono, correct device)
        audio = sd.rec(
            int(0.1 * SAMPLE_RATE_VAD),
            samplerate=SAMPLE_RATE_VAD,
            channels=1,
            dtype='float32',
            blocking=True
        )
        audio = audio.flatten()  # ensure 1D
        buffer.append(audio)

        # VAD check
        speaking = vad_detect(audio)

        if speaking:
            silence_start = None
        else:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > threshold_silence:
                break

    print("🎤 Finished listening.")
    buffer = np.concatenate(buffer)
    return buffer


#5. Timer-Based Listening 


sd.default.device = (0, 1)        # (input mic, output speakers)
sd.default.samplerate = 16000     # Whisper-compatible

def listen_fixed(seconds=3):
    print(f"\n🎤 Listening for {seconds} seconds...")

    audio = sd.rec(
        int(seconds * 16000),
        samplerate=16000,
        channels=1,
        dtype='float32',
        blocking=True
    )

    audio = audio.flatten()  # ensure 1D
    print("🎤 Done listening.")

    return audio




# 6. Transcribe Speech → Text  (Working Whisper Code)


def transcribe(audio):
    print("📝 Transcribing speech...")

    audio = np.array(audio).astype(np.float32)

    result = whisper_model.transcribe(
        audio,
        language="en",
        task="transcribe",
        fp16=False
    )

    text = result["text"].strip()
    print("You said:", text)
    return text


import re

def split_into_segments(text, max_chars=300):
    """
    Split text into safe chunks for Silero TTS.
    max_chars = recommended limit (200–350).
    """
    # First split by sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks


# 7. Speak with Silero

def speak(text, speaker=DEFAULT_SPEAKER, sr=SAMPLE_RATE):
    chunks = split_into_segments(text)

    for chunk in chunks:
        print(f"🔊 Speaking chunk: {chunk[:50]}...")
        audio = tts_model.apply_tts(
            text=chunk,
            speaker=speaker,
            sample_rate=sr
        )
        audio_np = np.array(audio)

        sd.play(audio_np, sr)
        sd.wait()





# 8. Main Loop 

if __name__ == "__main__":
    print("\n🤖 READY — Say something anytime.\n")

    while True:

        audio = listen_until_silence()
        text = transcribe(audio)

        if text == "":
            print("No speech detected.")
            continue

        if text.lower() in ["exit", "quit", "stop"]:
            speak("Shutting down. Goodbye.")
            break

        reply = llm_response(text)
        speak(reply)

