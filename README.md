# AI Sales Assistant Chatbot

## Project Goal

This project aims to create a realistic, low-latency chatbot that functions as an AI sales assistant for Nooks, an AI-powered sales development platform.
The chatbot responds when the user falls silent for some time, simulating a natural conversation flow.

## Current Implementation

The system consists of three main components:

1. Speech-to-text (STT) using NeMo for real-time transcription
2. A sales chatbot powered by OpenAI's GPT-4 model
3. Text-to-speech (TTS) using ElevenLabs for voice output

The chatbot listens to user input, transcribes it in real-time, and generates a response when the user stops speaking. The AI's response is then converted to speech and played back to the user.

## Task

Assume that you are not allowed to modify the base models used (you must use Nvidia's FastConformer model for STT, OpenAI's GPT-4 for the chatbot, and ElevenLabs with this voice setting for TTS).
How would you modify the code to make the chatbot lower latency & respond faster?

### Evaluation Criteria

Your solution will be evaluated based on:

1. Reduction in overall latency
2. Maintenance of conversation quality and realism (i.e the chatbot doesn't interrupt the human speaker while they're in the middle of speaking)
3. Code quality and clarity of explanation

## Getting Started

1. Review the existing code in `main.py`, `lib/sales_chatbot.py`, and `lib/elevenlabs_tts.py`
2. Install the requirements by running `pip install -r requirements.txt` (or use a virtual environment if you prefer)
3. Run the current implementation to understand its behavior by running `python main.py`
4. Begin your optimization process. Document your changes and reasoning in this README.md file when done.

## Poetry Setup

If you're getting stuck with installation issues, we offer an alternative Poetry-based installation method. 
1. Install [Poetry](https://python-poetry.org/docs/#installing-with-pipx)
2. Install all requirements by running `poetry install`
3. Run the current implementation by running `poetry run python3 main.py`

Good luck!

## Bonus Extensions

Right now when the chatbot is generating a response, even if users speak and try to interrupt the chatbot, the chatbot will talk over the user and not register what they're saying.
This isn't realistic - humans usually don't continue to talk when interrupted. How can we implement more realistic conversational behavior for the chatbot when it is interrupted?
