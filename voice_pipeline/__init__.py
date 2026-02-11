"""
Reusable building blocks for the voice → text → LLM → speech pipeline.

Modules:
- audio_input: microphone recording helpers
- stt_whisper: Whisper-based speech-to-text wrapper
- text_cleaning: simple text normalization utilities
- llm_client: OpenAI chat client + conversation memory
- tts_output: OpenAI TTS helpers (local playback / synth to file)
- reachy_connector: Reachy-specific audio playback utilities
"""

