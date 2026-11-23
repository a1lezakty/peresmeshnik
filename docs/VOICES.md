# Your voice and the cloud

The details behind the short sections of the [README](../README.md#your-voice). The other person's voice
("them → me") is in the README's [Languages](../README.md#languages) section.

## Your cloned voice

The app clones **your own voice only**.

- **The check.** A recording (or an imported file) speaks for you once you read two random phrases at the microphone
  and the app hears those words in the voice of every part of the recording, the start of the recording (what the
  clone takes) in the voice of the rest of it, and the clone made from it in your live voice.
- After three failed tries it waits a while.
- This stops a downloaded recording of someone else from speaking for you in a few clicks; it cannot stop someone who
  edits the app's files.
- An earlier build's voice is confirmed once the same way.
- **What is kept.** The live reading's speaker vector (256 numbers, not its sound) is kept in the voice's `voice.json`
  and goes with the recording to your own server.
- Tray → **Voices…** lists and deletes the voices.
- The clone is made from a 30–60 s recording.

**Where it runs.**

- **On any computer, a Mac too:** Kyutai's Pocket TTS on the CPU in English, French, German, Spanish, Portuguese,
  Italian and Russian.
  - Its weights (all but Russian) need your Hugging Face token (type Read), of an account that accepted Kyutai's
    conditions on huggingface.co/kyutai/pocket-tts.
  - Paste it in the wizard's Components or "Your voice" page or in Voices…; saved empty, it is forgotten.
- **On an NVIDIA GPU with 6 GB or more** (Windows, Linux; on Windows also a recent driver, see
  [the Windows voice pack](MODELS.md#the-windows-voice-pack)): XTTS-v2 in 13 languages.
- **Elsewhere** a stock voice or the cloud speaks for you.

## The cloud

**The cloud is Google's, with your own key.**

- A direction set to the cloud sends its sound to Google Gemini Live: the other person's voice for "them → me", yours
  for "me → them".
- It needs your own Gemini API key (aistudio.google.com → Get API key).
- On the free tier Google may use that audio to improve its models; a paid key is billed by Google, about $2.2 an hour
  per direction ([ARCHITECTURE §9](ARCHITECTURE.md#9-hardware-tiers)).
- Gemini is a simultaneous interpreter: its translation lags ~3–4 s behind the speech, and its voice is not yours.

**Which direction runs where.**

- On a Mac with 16 GB or more, once your cloned voice can speak their language there (Pocket TTS: your recorded voice,
  its models downloaded, for most languages the Hugging Face token above), "me → them" is on this computer unless you
  chose otherwise: your voice starts ~1.5 s after each phrase ([BENCH](BENCH.md#032-the-delay)).
- Elsewhere without an NVIDIA GPU the wizard suggests the cloud for what you say; a direction whose languages have
  local models (the [README's table](../README.md#languages)) can stay on this computer instead (a stock voice then
  speaks for you).
- With both directions here no key is needed and nothing is sent.
