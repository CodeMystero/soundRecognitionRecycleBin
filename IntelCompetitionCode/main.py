import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import time
import numpy as np
import librosa
import os
import cv2
import openvino as ov
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from argparse import SUPPRESS, ArgumentParser
from pathlib import Path
import threading
from queue import Empty, Queue
import pyaudio
import wave
import copy
import subprocess

import motor_test as mt

stepMotor_script_path = "step_motor.py"

def build_argparser():
    """Parses command line arguments."""
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group("Options")
    args.add_argument(
        "-h",
        "--help",
        action="help",
        default=SUPPRESS,
        help="Show this help message and exit.",
    )
    args.add_argument(
        "-i",
        "--input",
        required=False,
        help="Required. An input to process. The input must be a single image, "
        "a folder of images, video file or camera id.",
    )
    args.add_argument(
        "-m",
        "--models",
        type=Path,
    )
    args.add_argument(
        "-it",
        "--inference_type",
        help="Optional. Type of inference for single model.",
        choices=["sync", "async"],
        default="sync",
        type=str,
    )
    args.add_argument(
        "-l",
        "--loop",
        help="Optional. Enable reading the input in a loop.",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "--no_show",
        help="Optional. Disables showing inference results on UI.",
        default=False,
        action="store_true",
    )
    args.add_argument(
        "-d",
        "--device",
        help="Optional. Device to infer the model.",
        choices=["CPU", "GPU"],
        default="CPU",
        type=str,
    )
    args.add_argument(
        "--output",
        default=None,
        type=str,
        help="Optional. Output path to save input data with predictions.",
    )
    args.add_argument(
        "-a",
        "--audio",
        type=Path
    )

    return parser

def get_inferencer_class(type_inference, models):
    """Return class for inference of models."""
    if len(models) > 1:
        type_inference = "chain"
        print("You started the task chain pipeline with the provided models in the order in which they were specified")
    return EXECUTORS[type_inference]

def Therad_recording(audio_queue:Queue, lock)->None:
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5

    Pyaud = pyaudio.PyAudio()
    stream = Pyaud.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

    RECORD_FORCE_STOP = False
    print("Start recording")
    while not RECORD_FORCE_STOP:
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            data = np.frombuffer(data, dtype=np.float32)
            frames.extend(data)
        with lock:
            audio_queue.put( ("wav", frames) )

    stream.close()

def Therad_audio_infer(audio_queue:Queue, spec_audio_queue:Queue, lock)->None:
    """Main function that is used to run demo."""

    args = build_argparser().parse_args()
    Classes_Name = ['can', 'others', 'paper', 'plastic']

    core = ov.Core()
    model = core.read_model(model=args.models)
    compiled_model = core.compile_model(model=model, device_name=args.device)

    input_layer_ir = compiled_model.input(0)
    output_layer_ir = compiled_model.output(0)
    N, C, H, W = input_layer_ir.shape

    SOUND_DURATION = 5
    sample_rate = 44100 # Hz
    HOP_LENGTH = 1024        # number of samples between successive frames
    WINDOW_LENGTH = 1024     # length of the window in samples
    N_MEL = 256             # number of Mel bands to generate

    Force_stop = False
    while not Force_stop:
        try:
            with lock:
                event = audio_queue.get_nowait()
                audio_queue.task_done()
        except Empty:
            continue
        name, frame = event

        if name == "wav":
            frame = np.array(frame)
            max_v = np.max(np.abs(frame))
            if max_v < 0.9999:  # Skip frames with volume below threshold
                continue
            with lock:
                melspectrogram = librosa.feature.melspectrogram(y=frame,
                sr=sample_rate,
                hop_length=HOP_LENGTH,
                win_length=WINDOW_LENGTH,
                n_mels=N_MEL)
            melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
            norm = mpl.colors.Normalize(vmin = -80, vmax = 0)
            melcmap = librosa.display.cmap(melspectrogram_db)
            ColorMapping = mpl.cm.ScalarMappable(norm=norm, cmap=melcmap)
            melspectrogram_db_map = ColorMapping.to_rgba(melspectrogram_db)
            melspectrogram_db_map = melspectrogram_db_map.astype('float32')
            melspectrogram_db_map_RGB = cv2.cvtColor(melspectrogram_db_map, cv2.COLOR_BGRA2RGB)
            melspectrogram_db_map_RGB_flip = np.flip(melspectrogram_db_map_RGB, axis = 0)
            resized_image = cv2.resize(melspectrogram_db_map_RGB_flip, (W, H))
            resized_image = resized_image*255
            resized_image_show = cv2.resize(melspectrogram_db_map_RGB_flip, (2*W, 2*H))
            que_img = copy.deepcopy(resized_image_show)
            input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
            
            start_time = time.time()  # Record inference start time
            results = compiled_model([input_image])[output_layer_ir]
            inference_time = time.time() - start_time  # Calculate inference time

            result_label = Classes_Name[np.argmax(results)]
            spec_audio_queue.put((name, frame, result_label, inference_time))

def update_gui(root, image_label, time_label,detection_label, spec_audio_queue, image_size, lock, threshold, current_bin):
    try:
        with lock:
            event = spec_audio_queue.get_nowait()
            spec_audio_queue.task_done()
            name, frame, result, inference_time = event
            image_path = f"images/{result}.jpg"
            image = Image.open(image_path).resize(image_size)
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
            time_label.config(text=f"Inference time: {inference_time:.4f} seconds")
            detection_label.config(text=f"{result.upper()} DETECTED")
            # Schedule motor control to run after GUI update
            root.after(100, control_motor, result, frame, threshold, current_bin)
            current_bin = result
            root.after(200, reset_gui, image_label, time_label, detection_label, "images/main.jpg",image_size)
    except Empty:
        pass
    root.after(1000, update_gui, root, image_label, time_label,detection_label, spec_audio_queue, image_size, lock, threshold, current_bin)

def reset_gui(image_label, time_label, detection_label, initial_image_path, image_size):
    initial_image = Image.open(initial_image_path).resize(image_size)
    initial_photo = ImageTk.PhotoImage(initial_image)
    image_label.config(image=initial_photo)
    image_label.image = initial_photo
    time_label.config(text="Inference time: 0.0000 seconds")
    detection_label.config(text="Waiting for detection...")

def control_motor(result, frame, threshold, current_bin):
    max_v = np.max(np.abs(frame))
    if result and (max_v > threshold):
        #print("Detected: ", result)
        target_bin = result
        #print("TargetBin: ",target_bin)
        #print("CurrentBin: ",current_bin)
        mt.rotating_bin(mt.dist_(current_bin, target_bin))
        subprocess.run(["python3", stepMotor_script_path])
        #current_bin = target_bin

#current_bin = 'paper'
#print("CurrentBin_Wrong Init:",current_bin)
def main():
    args = build_argparser().parse_args()
    if args.loop and args.output:
        raise ValueError("--loop and --output cannot be both specified")

    audio_queue = Queue()
    spec_audio_queue = Queue()
    lock = threading.Lock()
    Thread_audio_play = threading.Thread(target=Therad_audio_infer, args=(audio_queue, spec_audio_queue, lock))
    Thread_mic_record = threading.Thread(target=Therad_recording, args=(audio_queue, lock))

    Thread_audio_play.start()
    Thread_mic_record.start()
    FORCE_STOP = False

    threshold = 0.9999

    current_bin ='others'
    #target_bin = 'others'

    image_size = (2548, 1150)  # Fixed image size

    root = tk.Tk()
    root.title("Incheon HRD Center Team: EcoSort")

    title_label = tk.Label(root, text="Incheon HRD Center Team: EcoSort", font=("Helvetica", 25))
    title_label.pack(pady=10)

    initial_image_path = "images/main.jpg"
    initial_image = Image.open(initial_image_path).resize(image_size)
    initial_photo = ImageTk.PhotoImage(initial_image)
    image_label = Label(root, image=initial_photo)
    image_label.pack(pady=10)

    time_label = tk.Label(root, text="Inference time: 0.0000 seconds", font=("Helvetica", 40))
    time_label.pack(side="right", padx=10, pady=20)

    detection_label = tk.Label(root, text="Waiting for detection ...", font = ("Helvetica",50))
    detection_label.pack(side ="left", padx = 10, pady = 20)

    root.after(1000, update_gui, root, image_label, time_label,detection_label, spec_audio_queue, image_size, lock, threshold, current_bin)
    root.mainloop()

    Thread_audio_play.join()
    Thread_mic_record.join()

if __name__ == "__main__":
    sys.exit(main() or 0)
