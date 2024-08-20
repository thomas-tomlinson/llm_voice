import time
import threading
import json
import numpy as np
import sounddevice as sd
from queue import Queue
from lightning_whisper_mlx import LightningWhisperMLX
from rich.console import Console
import llm 
import ringbuffer
import webrtcvad
import sh

WAKE_WORD="computer"
SAMPLE_RATE=16000
INTERVAL_SIZE_MS = 30 
BLOCK_SIZE = int(SAMPLE_RATE * INTERVAL_SIZE_MS / 1000)

console = Console()
whisper = LightningWhisperMLX(model="base", batch_size=12, quant=None)
llm = llm.Llm()
say_command = sh.Command('say')
data_queue = Queue()
vad = webrtcvad.Vad()

def audio_callback(indata, frames, time, status):
    if status:
        console.print(status)
    data_queue.put(bytes(indata))

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)

def transcribe_audio(audio_chunk):
    """this takes a byte object of audio and transcribes it"""
    audio_np = (
        np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    )
    if audio_np.size < 0:
        return None

    result = whisper.transcribe(audio_np)  # Set fp16=True if using a GPU
    return result

def process_audio(stop_event, data_queue):
    """realtime decoding from the queue"""
    vad_count = 0
    ring_buffer = ringbuffer.RingBuffer(50)  # 30 * 30ms = 600ms ring buffer
    while not stop_event.is_set():
        # we loop through looking for frames of audio with a VAD postive 
        # if we find one, wait until we have enough to look for the wake word

        audio_frame = data_queue.get()
        ring_buffer.add(audio_frame)
        if vad.is_speech(audio_frame, SAMPLE_RATE):
            vad_count += 1

        if vad_count > 10:
            #look for the wake word
            audio_data = b''
            audio_data = audio_data.join(ring_buffer.get())
            text = transcribe_audio(audio_data)
            console.print(f"[green]wake word debug: {text}")
            console.print(f"[green]ring length  {len(ring_buffer.get())}")
            if WAKE_WORD in text['text'].lower():
                console.print("[yellow]Wake word detected")
                user_input = wake_word_processing(audio_data, data_queue) 
                stream.stop()
                llm_execute(user_input['text'])
                stream.start()
                ring_buffer.clear()
            vad_count = 0

def llm_execute(prompt):
    llm_response = llm.chat(prompt)
    console.print(f"[red] {llm_response}")
    say_command(llm_response)

def wake_word_processing(audio_chunk, data_queue):
    """wait unitl we believe speaking has stopped and process the user input"""
    empty_vads = 0
    while empty_vads < 5:
        new_audio = data_queue.get()
        audio_chunk += new_audio

        speech = vad.is_speech(new_audio, SAMPLE_RATE)
        if speech is False:
            empty_vads += 1

    prompted_text = transcribe_audio(audio_chunk)
    console.print(f"[red]User Prompt: {prompted_text['text']}")
    return prompted_text

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                dtype="int16",
                channels=1,
                callback=audio_callback,
                blocksize=BLOCK_SIZE
            )

            stop_event = threading.Event()
            stream.start()
            processing_thread = threading.Thread(
                target=process_audio,
                args=(stop_event, data_queue),
            )
            processing_thread.start()

            input()
            stop_event.set()
            time.sleep(0.1)
            stream.stop()
            processing_thread.join()


    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")