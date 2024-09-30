import os
import time
import pyaudio as pa
import numpy as np
import threading
from packages.sales_chatbot import SalesChatbot
from packages.nemo_stt import StreamingTranscription  # Update the class name if different
from packages.elevenlabs_tts import speak, generate_audio_task  # Update the class name if different
from elevenlabs import play

SAMPLE_RATE = 16000
CHUNK_SIZE = 160  # ms
WAIT_TIME = 500  # ms
RESPONSE_TIMEOUT = 4  # seconds for AI response
SENTENCE_MIN_LENGTH = 3  # minimum character length of a sentence to be considered valid
PREMADE_SENTENCES = [
    "Please give me a moment to respond.",
    "Let me get that answer for you.",
    "Please Give me a second to come up with the best response.",
]

PREMADE_AUDIOS = []

transcriber = StreamingTranscription()
chatbot = SalesChatbot()

state = {
    "last_text": "",
    "silence_duration": 0,
}

for premade_sentence in enumerate(PREMADE_SENTENCES):
    idx, audio = generate_audio_task(premade_sentence)
    PREMADE_AUDIOS.append(audio)

# Create an Event object to signal the process_response thread
terminate_event_list = []
# The signal that indicate the premade sentence is done playing, so the AI can speak
premade_sentence_done_event = threading.Event()
# The signal that indicate the conversation is finished
conversation_finished_event = threading.Event()

# Function to play a premade sentence
def play_premade_sentence():
    random_idx = np.random.randint(0, len(PREMADE_AUDIOS))
    print(f"AI: {PREMADE_SENTENCES[random_idx]}")
    play(PREMADE_AUDIOS[random_idx])
    premade_sentence_done_event.set()

# Function to handle AI response generation and speaking in a separate thread
def process_response(user_text, terminate_event):
    print(f"USER: {user_text}")
    
    premade_sentence_done_event.clear()

    # Create a timer to monitor response time
    timer_thread = threading.Timer(RESPONSE_TIMEOUT, play_premade_sentence)
    timer_thread.start()  # Start the timer for 5 seconds
    
    # Simulate response generation and speaking, while checking for the terminate signal
    if not terminate_event.is_set():
        ai_response = chatbot.generate_response(user_text)
        if ai_response["type"] == "reservation":
            response = chatbot.reserve_demo(ai_response["time"], ai_response["email"])
            if response["status"] != "success":
                ai_response["messages"] = "I'm sorry, I couldn't schedule the demo. Please try again later."
        timer_thread.cancel()  # Cancel the premade sentence if the response is ready in time
        premade_sentence_done_event.set()

        print(f"AI: {ai_response}")
        speak([response for response in ai_response["messages"] if len(response) >= SENTENCE_MIN_LENGTH], terminate_event, premade_sentence_done_event)
        if ai_response["type"] == "end":
            print("Terminating the conversation...")
            conversation_finished_event.set()


def callback(in_data, *_):
    signal = np.frombuffer(in_data, dtype=np.int16)
    text = transcriber.transcribe_chunk(signal)
    # print(f"Transcribed: {text}")
    if text != state["last_text"]:
        state["last_text"] = text
        state["silence_duration"] = 0
    else:
        state["silence_duration"] += CHUNK_SIZE / 2

        if (len(state["last_text"]) > 0):
            # as soon as the user starts speaking, the AI should stop speaking
            for event in terminate_event_list:
                event.set()
        # Check if the user said anything and at least WAIT_TIME has since passed
        if state["silence_duration"] >= WAIT_TIME and len(state["last_text"]) > 0:
            for event in terminate_event_list:
                event.set()
            new_event = threading.Event()
            response_thread = threading.Thread(target=process_response, args=(state["last_text"], new_event))
            response_thread.start()
            terminate_event_list.append(new_event)
            
            # Reset transcription cache and keep transcribing
            transcriber.reset_transcription_cache() 
            state["last_text"] = ""

    # Transcription processing is paused while the AI response is being generated, continue now
    return (in_data, pa.paContinue)

p = pa.PyAudio()
print('Available audio input devices:')
input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        print(i, dev.get('name'))

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        print('Please type input device ID:')
        dev_idx = int(input())

    stream = p.open(format=pa.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=dev_idx,
        stream_callback=callback,
        frames_per_buffer=int(SAMPLE_RATE * CHUNK_SIZE / 1000) - 1
    )

    print('Listening...')

    stream.start_stream()
    
    new_event = threading.Event()
    speak(["Hello this is Nooks. How can I help you today?"], new_event)
    terminate_event_list.append(new_event)
    
    try:
        while stream.is_active() and not conversation_finished_event.is_set():
            time.sleep(0.1)
    finally:        
        stream.stop_stream()
        stream.close()
        p.terminate()

        print()
        print("PyAudio stopped")
else:
    print('ERROR: No audio input device found.')
