# AI Sales Assistant Chatbot

## Highlights

1. I change the sales chatbot prompt to respond in json format and support various functions. I use in-context learning to improve the chatbot's performance (because the current task is not very complex, I don't need to use chain-of-thought for now). Further improvements can be made by adding RAG to the chatbot, including the real bussiness senarios and previous customer interactions to improve the chatbot's personalization and performance.

2. I made the chatbot able to respond multiple sentences at a time, and to optimize the tts performance, I generate the audio of all sentences in background and multiple threads and play them in sequence. This can improve the response speed of the chatbot.

3. To support the interruption of the chatbot, I use a threading signal to control the chatbot's response. When the chatbot is interrupted, the chatbot will stop responding and wait for the next command. This can improve the user experience of the chatbot.

4. I added a preamble to inform the user that the nooks chatbot is ready to respond. This can improve the user experience of interacting with the chatbot.

5. To cope with the unexpected latency of the gpt or tts response time, I will randomly play some of the pre-generated audio to the user to inform the user that the chatbot is still processing the response. Also, to avoid the chatbot start responding when the pre-generated audio is playing, I use a threading signal to control the chatbot's response.

6. I changed the play from ffmpeg to pydub, which can improve the audio playing performance and reduce the latency of the audio playing. but please `pip install sounddevice soundfile` first. I have added them to the requirements.txt.

7. I supported the auto demo book which will call the `chatbot.reserve_demo()` function once the chatbot detects the user is interested in the demo and provides some contact and time information. I also support the auto termination of the chatbot after the end of conversation is detected.

## Further Improvements

1. For the acoustic echo cancellation problem, I suggest using py-webrtc as the input audio processing library. If you test without a headphone, the echo of output audio will be captured by the input microphone, and the echo will be fed back to the output audio. This will cause a feedback loop and the output audio will be distorted. So I highly recommend using a headphone when testing the audio processing algorithm or using an acoustic echo cancellation algorithm to remove the echo from the output audio.

2. Modify the play function of elevenlabs package to support shorter interval between the sentences. Because the current play function will wait for the previous audio to finish before playing the next audio, this will cause a delay between the sentences.

## Short Demo

Here is a short demo video of the chatbot:
<video width="720" height="480" controls>

  <source src="assets/demo_video.mp4" type="video/mp4">
  Your browser does not support the video tag. Please refer to the video in the assets folder directly.
</video>
