import argparse
import base64
import sys
import time
import cv2
from collections import deque
from picamera2 import Picamera2
from openai import OpenAI
from pydantic import BaseModel, Field


class FrameText(BaseModel):
    text: str = Field(description="New text visible in the current frame, not yet captured")


# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--logs', action='store_true')
args = parser.parse_args()

def log(msg):
    if args.logs:
        print(f"[{msg}]", file=sys.stderr, flush=True)


# ── camera setup (once) ───────────────────────────────────────────────────────

picam = Picamera2()
picam.configure(picam.create_still_configuration(main={"size": (4608, 2592)}))
picam.start()
time.sleep(1)


# ── openai ────────────────────────────────────────────────────────────────────

client = OpenAI()  # reads OPENAI_API_KEY from env

history_frames = deque(maxlen=5)  # last 5 frames for context
recognized = []                   # text segments appended per frame


# ── helpers ───────────────────────────────────────────────────────────────────

def capture():
    frame = picam.capture_array()
    cv2.imwrite("photo.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    log("Photo taken")
    return frame

def to_base64(frame):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode('utf-8')

def read_frame(frame):
    log("Sending to OpenAI")

    b64 = to_base64(frame)
    history_frames.append(b64)

    content = []
    for img in history_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img}"}
        })

    already_read = ''.join(recognized)
    context = (
        f"Text captured so far (left to right): {already_read!r}. "
        "The rightmost characters of this string may overlap with what you see in the current frame — "
        "output only the portion that comes after what has already been captured. "
    ) if recognized else ""

    content.append({
        "type": "text",
        "text": (
            "You are an OCR assistant reading printed text by scanning it from left to right, "
            "one camera frame at a time. The camera is panning slowly, so consecutive frames overlap slightly. "
            "The images are shown in chronological order; the LAST image is the current frame to read. "
            f"{context}"
            "The images are the source of truth — previous OCR passes may have made mistakes, "
            "so do not let the already-captured text override what you clearly see in the image. "
            "If a character is ambiguous, use surrounding context (visible word shapes, spacing) to make "
            "your best guess, but prefer what the image shows over what 'should' come next. "
            "Output only the new text from the current frame (excluding any overlap already captured), "
            "preserving spaces and punctuation exactly as they appear."
        )
    })

    response = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        max_completion_tokens=200,
        messages=[{"role": "user", "content": content}],
        response_format=FrameText
    )
    text = response.choices[0].message.parsed.text
    log(f"Got: {repr(text)}")
    return text


# ── main loop ─────────────────────────────────────────────────────────────────

log("Starting — Ctrl+C to stop")
try:
    while True:
        frame = capture()
        text = read_frame(frame)
        if text:
            recognized.append(text)
            print(text, end='', flush=True)

except KeyboardInterrupt:
    print()  # newline when done
    picam.stop()
