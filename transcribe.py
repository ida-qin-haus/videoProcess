#!/usr/bin/env python3
"""
transcribe.py - Transcribe a screen recording of scrolling code to a source file.

Usage:
    python transcribe.py <video.mov> [-o output.R] [--fps 3] [--model claude-sonnet-4-6]

Requirements:
    pip install opencv-python anthropic
    ANTHROPIC_API_KEY environment variable must be set.
"""

import sys
import os
import argparse
import base64
import cv2
import numpy as np
from pathlib import Path
import anthropic


PROMPT = """\
You are transcribing a screen recording of a source code file being scrolled from top to bottom \
in a browser or editor.

The images below are sequential frames (chronological order) from the recording. Each frame shows \
a portion of the code. As the recording progresses, the view scrolls down, revealing content \
further down in the file.

Your task: reconstruct the COMPLETE, VERBATIM source code file.

Rules:
- Output ONLY the raw source code. No markdown code fences. No commentary. No preamble.
- Preserve all indentation, whitespace, and line breaks exactly as shown.
- Consecutive frames overlap — combine them to get full coverage; do not duplicate lines.
- If a dialog box, overlay, or UI element obscures code, read what is visible around it. \
Mark any line that cannot be recovered as: # UNREADABLE
- Lines may be cut off at the right edge of the frame — use surrounding context to complete \
obvious truncations where possible.
- Output every line from the very first line to the very last line visible across all frames.
"""


def extract_frames(video_path: str, fps: float, min_diff: float) -> list[bytes]:
    """Extract distinct frames from video, skipping near-identical ones."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(video_fps / fps))

    print(f"Video: {video_fps:.0f} fps, {total_frames} frames ({total_frames / video_fps:.1f}s) "
          f"— sampling every {step} frames ({fps} fps)")

    frames: list[bytes] = []
    prev_gray = None

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
            if diff < min_diff:
                continue

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frames.append(buf.tobytes())
        prev_gray = gray

    cap.release()
    print(f"Kept {len(frames)} distinct frames after deduplication")
    return frames


def transcribe(frames: list[bytes], model: str) -> str:
    """Send frames to Claude and return the transcribed code."""
    client = anthropic.Anthropic()

    content: list[dict] = [{"type": "text", "text": PROMPT}]
    for frame_bytes in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(frame_bytes).decode(),
            },
        })

    print(f"Sending {len(frames)} frames to {model} ...")
    response = client.messages.create(
        model=model,
        max_tokens=8096,
        messages=[{"role": "user", "content": content}],
    )

    code = response.content[0].text
    # Strip accidental markdown fences if the model added them despite instructions
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def detect_extension(code: str) -> str:
    head = code[:200]
    if "Rscript" in head or "<-" in head:
        return ".R"
    if "python" in head.lower() or head.lstrip().startswith(("import ", "from ", "def ", "class ")):
        return ".py"
    return ".txt"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe a screen recording of scrolling code to a source file."
    )
    parser.add_argument("video", help="Path to video file (.mov, .mp4, …)")
    parser.add_argument("-o", "--output", help="Output file path (auto-detected if omitted)")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="Sampling rate in frames per second (default: 3)")
    parser.add_argument("--min-diff", type=float, default=0.02,
                        help="Minimum mean pixel difference to keep a frame (default: 0.02)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model (default: claude-sonnet-4-6)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set", file=sys.stderr)
        sys.exit(1)

    frames = extract_frames(str(video_path), fps=args.fps, min_diff=args.min_diff)
    if not frames:
        print("Error: no frames extracted", file=sys.stderr)
        sys.exit(1)

    code = transcribe(frames, model=args.model)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = video_path.with_suffix(detect_extension(code))

    out_path.write_text(code)
    print(f"Written → {out_path}")


if __name__ == "__main__":
    main()
