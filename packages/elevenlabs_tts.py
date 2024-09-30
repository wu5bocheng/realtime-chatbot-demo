import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from elevenlabs import play
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv()

# Get API key from environment
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # Default voice, you can change this

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def generate_audio_task(idx_text):
    '''
    This function is responsible for generating audio for a given text input
    idx_text is a tuple where idx is the index and text is the text to be processed
    '''
    idx, text = idx_text
    try:
        # Generate the audio using the ElevenLabs API
        audio = client.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_monolingual_v1"
        )
        return idx, audio
    except Exception as e:
        print(f"Error generating audio for text at index {idx}: {str(e)}")
        # Return None if an error occurs
        return idx, None

def speak(texts, terminate_event = None, premade_sentence_done_event = None):
    '''
    This function orchestrates the process of generating and playing audio sequentially
    `texts` is the list of texts to be converted into speech, and `terminate_event` is used to handle termination
    '''
    print(f"Speaking... {texts}")
    # Buffer to store audio clips, keyed by their index for sequential playback
    buffer = {}
    # Track the index of the next expected audio clip for sequential playback
    expected_idx = 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        # Create a list of (index, text) pairs to pass to each worker
        idx_texts = list(enumerate(texts))

        # Submit audio generation tasks to the executor and store the future objects
        # `future_to_idx` maps future objects to the index of the corresponding text
        future_to_idx = {executor.submit(generate_audio_task, idx_text): idx_text[0] for idx_text in idx_texts}

        try:
            for future in as_completed(future_to_idx):
                if terminate_event and terminate_event.is_set():
                    print("Termination signal received. Stopping playback.")
                    break

                idx = future_to_idx[future]
                idx, audio = future.result()

                if audio is not None:
                    buffer[idx] = audio
                    # Play audio clips in sequential order based on `expected_idx`
                    # If the audio for the expected index is available, play it
        except Exception as e:
            # Handle any exceptions that occur during playback
            print(f"Error during playback: {str(e)}")

    while expected_idx in buffer:
        if terminate_event and terminate_event.is_set():
            print("Termination signal received during playback.")
            break

        # wait for the premade sentence to finish playing before playing the next clip
        if premade_sentence_done_event and not premade_sentence_done_event.is_set():
            premade_sentence_done_event.wait()

        play(buffer[expected_idx], use_ffmpeg=False)
        del buffer[expected_idx]

        # Increment the expected index to play the next clip
        expected_idx += 1