import argparse
import base64
import sys
import time
import threading
import cv2
from collections import deque
from picamera2 import Picamera2
from openai import OpenAI
from pydantic import BaseModel, Field


class FrameText(BaseModel):
    text: str = Field(description="New text visible in the frames, not yet captured")


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--logs', action='store_true')
args = parser.parse_args()

def log(msg):
    if args.logs:
        print(f"[{msg}]", file=sys.stderr, flush=True)


# ── constants ─────────────────────────────────────────────────────────────────

CAPTURE_FPS     = 1
FRAME_BUF_SIZE  = 15
LLM_SIZE        = (1536, 864)   # resize before sending; capture stays at full res


# ── camera setup ──────────────────────────────────────────────────────────────

picam = Picamera2()
picam.configure(picam.create_still_configuration(main={"size": (4608, 2592)}))
picam.start()
time.sleep(1)


# ── shared state ──────────────────────────────────────────────────────────────

client = OpenAI()  # reads OPENAI_API_KEY from env

frame_buffer   = deque(maxlen=FRAME_BUF_SIZE)  # rolling window of resized base64 frames
frame_lock     = threading.Lock()
new_frame_evt  = threading.Event()             # signals LLM thread that at least one new frame arrived
recognized     = []                            # text segments in order (only written by LLM thread)
running        = True


# ── helpers ───────────────────────────────────────────────────────────────────

def to_base64_small(frame):
    """Convert full-res RGB frame → resized JPEG → base64 string."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized   = cv2.resize(frame_bgr, LLM_SIZE, interpolation=cv2.INTER_AREA)
    _, buf    = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')


# ── camera thread ─────────────────────────────────────────────────────────────

def camera_loop():
    interval = 1.0 / CAPTURE_FPS
    while running:
        t     = time.time()
        frame = picam.capture_array()
        log("Frame captured")
        b64   = to_base64_small(frame)
        with frame_lock:
            frame_buffer.append(b64)
        new_frame_evt.set()                     # wake LLM thread
        elapsed = time.time() - t
        time.sleep(max(0.0, interval - elapsed))


# ── LLM thread ────────────────────────────────────────────────────────────────

def read_frames(frames_snapshot):
    content = []
    for img in frames_snapshot:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    already_read = ''.join(recognized)

    if already_read:
        context_block = (
            f"TEXT CAPTURED SO FAR:\n{already_read}\n\n"
            "Continue from exactly where this ends. "
            "The captured text is a best-effort record — it may contain OCR errors or noise. "
            "If what you see in the frames clearly contradicts the tail of the captured text, "
            "trust the images and continue from the correct point anyway.\n\n"
        )
    else:
        context_block = "No text has been captured yet. Read whatever is visible in the frames.\n\n"

    content.append({
        "type": "text",
        "text": (
            "SETUP: You are an OCR assistant reading physical printed text. "
            "A Raspberry Pi camera is held by hand and moved slowly over the text — "
            "frames can be shaky, blurry, slightly rotated, or catch a transition between lines. "
            f"You receive a rolling buffer of the last {FRAME_BUF_SIZE} frames at {CAPTURE_FPS} FPS "
            f"(~{FRAME_BUF_SIZE // CAPTURE_FPS}s of footage). "
            "Frames are in chronological order, last frame is most recent. "
            "Oldest frames are dropped as new ones arrive — this is a sliding window, not the full scan. "
            "Images are captured at 4608×2592 and downscaled to 1536×864.\n\n"
            + context_block +
            "YOUR JOB: Output the new text visible in the frames that has not been captured yet. "
            "Use all the frames together to get the best read — a word that is blurry in one frame "
            "may be clearer in another. "
            "Do not repeat anything already in the captured text. "
            "Preserve spacing and punctuation exactly as visible. "
            "Only output an empty string if the frames contain no readable text at all."
        )
    })

    response = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        max_completion_tokens=200,
        messages=[{"role": "user", "content": content}],
        response_format=FrameText
    )
    return response.choices[0].message.parsed.text


def llm_loop():
    while running:
        # Block until camera signals at least one new frame (or timeout for clean shutdown)
        new_frame_evt.wait(timeout=1.0)
        new_frame_evt.clear()

        with frame_lock:
            frames_snapshot = list(frame_buffer)

        if not frames_snapshot:
            continue

        log(f"LLM call — {len(frames_snapshot)} frames in buffer")
        try:
            text = read_frames(frames_snapshot)
            log(f"Got: {repr(text)}")
            if text:
                recognized.append(text)
                print(text, end='', flush=True)
        except Exception as e:
            log(f"LLM error: {e}")


# ── start ─────────────────────────────────────────────────────────────────────

camera_thread = threading.Thread(target=camera_loop, daemon=True)
llm_thread    = threading.Thread(target=llm_loop,    daemon=True)

log("Starting — Ctrl+C to stop")
try:
    camera_thread.start()
    llm_thread.start()
    while True:
        time.sleep(0.1)

except KeyboardInterrupt:
    running = False
    print()
    picam.stop()
