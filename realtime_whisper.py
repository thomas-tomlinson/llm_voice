import time
import threading
import numpy as np
import sounddevice as sd
from queue import Queue
from lightning_whisper_mlx import LightningWhisperMLX
from rich.console import Console
import llm as llm
import sh

WAKE_WORD="computer"
SAMPLE_RATE=16000
INTERVAL_SIZE_MS = 100 
BLOCK_SIZE = int(SAMPLE_RATE * INTERVAL_SIZE_MS / 1000)

console = Console()
whisper = LightningWhisperMLX(model="base", batch_size=12, quant=None)
llm = llm.Llm()
say_command = sh.Command('say')
data_queue = Queue()

def audio_callback(indata, frames, time, status):
    if status:
        console.print(status)
    data_queue.put(bytes(indata))

def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.
    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.
    Returns:
        None
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

def transcribe_audio(stop_event, data_queue):
    """realtime decoding from the queue"""
    user_prompt_text = ''
    while not stop_event.is_set():
        qsize=data_queue.qsize()
        if qsize < 0:
            continue

        audio_data = b""
        for i in range(qsize):
            audio_data += data_queue.get()
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        if audio_np.size < 0:
            continue

        #with console.status("Transcribing...", spinner="earth"):
        #    text = transcribe(audio_np)
        #console.print(f"[yellow]You: {text}")
        text = transcribe(audio_np)
        console.print(f"[green]debug: queue_size: {qsize} whisper: {text}")
        if WAKE_WORD in text['text'].lower():
            #wake word detected
            #console.print(f"[yellow]You: {text['text']}")
            console.print("[yellow]Wake word detected")
            user_prompt_text += text['text']
            time.sleep(1)
            continue
        elif text['text'] != '' and user_prompt_text != '':
            user_prompt_text += text['text']
            time.sleep(1)
            continue
        else:
            if user_prompt_text != '':
                stream.stop()
                console.print(f"[yellow] {user_prompt_text}")
                llm_response = llm.chat(user_prompt_text)
                console.print(f"[red] {llm_response}")
                try:
                    text = llm_response['text']
                except TypeError:
                    text = llm_response
                say_command(text)
                stream.start()
                user_prompt_text = ''

        time.sleep(.5)

def transcribe(audio_np: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.
    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.
    Returns:
        str: The transcribed text.
    """
    #result = whisper.transcribe(audio_np, fp16=False)  # Set fp16=True if using a GPU
    result = whisper.transcribe(audio_np)  # Set fp16=True if using a GPU
    return result
    #text = result["text"].strip()
    #return text

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")

    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )

            #data_queue = Queue()  # type: ignore[var-annotated]
            stop_event = threading.Event()
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                dtype="int16",
                channels=1,
                callback=audio_callback,
                blocksize=BLOCK_SIZE
            )
            #recording_thread = threading.Thread(
            #    target=record_audio,
            #    args=(stop_event, data_queue),
            #)
            stream.start()
            processing_thread = threading.Thread(
                target=transcribe_audio,
                args=(stop_event, data_queue),
            )
            stream.start()
            processing_thread.start()

            input()
            stream.stop()
            stop_event.set()
            processing_thread.join()

            #audio_data = b"".join(list(data_queue.queue))
            #audio_np = (
            #    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            #)

            #if audio_np.size > 0:
            #    with console.status("Transcribing...", spinner="earth"):
            #        text = transcribe(audio_np)
            #    console.print(f"[yellow]You: {text}")

                #with console.status("Generating response...", spinner="earth"):
                #    response = get_llm_response(text)
                #    sample_rate, audio_array = tts.long_form_synthesize(response)

                #console.print(f"[cyan]Assistant: {response}")
                #play_audio(sample_rate, audio_array)
            #else:
            #    console.print(
            #        "[red]No audio recorded. Please ensure your microphone is working."
            #    )

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
