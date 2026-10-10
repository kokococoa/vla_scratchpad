import json
import os
import time

# OpenCV's Qt backend may look for fonts inside the cv2 wheel, where recent Qt
# builds no longer bundle them. Point it at the system DejaVu fonts instead.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
import torch
from PIL import Image
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from transformers import AutoProcessor, AutoModelForMultimodalLM

MODEL = "Qwen/Qwen3.5-2B"
IMAGE_PATH = "01_qwen_test/image/cargo.jpg"
OUTPUT_PATH = "01_qwen_test/image/cargo_annotated.jpg"
SHOW_PREVIEW = False
BBOX_COORDINATE_SCALE = 1000
PROMPT = """
Describe this scene.

Report:
1. Objects visible
2. Approximate spatial relationships
3. What task appears to be occurring
4. Anything relevant for a robot interacting with this scene
"""
PROMPT = """
How many red levers are there below the cargo?
Return only a JSON array. Each item must have a "bbox_2d" array of four integer
coordinates [x1, y1, x2, y2] and a "label" string. Normalize coordinates to a
0-1000 range, with the origin at the top-left. Return [] if none are visible.
Label each lever as "red lever N" where N is a 1-based index.
"""


def parse_detections(response):
    array_start = response.find("[")
    if array_start == -1:
        raise ValueError("Qwen response did not contain a JSON array.")

    detections, _ = json.JSONDecoder().raw_decode(response[array_start:])
    if not isinstance(detections, list):
        raise ValueError("Qwen JSON must be an array of detections.")

    for detection in detections:
        if not isinstance(detection, dict):
            raise ValueError("Each detection must be a JSON object.")
        bbox = detection.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Each detection must have four bbox_2d coordinates.")

    return detections


def draw_detections(image, detections):
    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    image_height, image_width = frame.shape[:2]

    for detection in detections:
        bbox = detection["bbox_2d"]
        normalized_x1, normalized_y1, normalized_x2, normalized_y2 = (
            float(coordinate) for coordinate in bbox
        )
        x1 = round(normalized_x1 * image_width / BBOX_COORDINATE_SCALE)
        y1 = round(normalized_y1 * image_height / BBOX_COORDINATE_SCALE)
        x2 = round(normalized_x2 * image_width / BBOX_COORDINATE_SCALE)
        y2 = round(normalized_y2 * image_height / BBOX_COORDINATE_SCALE)
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))
        if x1 >= x2 or y1 >= y2:
            continue

        label = str(detection.get("label", "object"))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return frame


processor = AutoProcessor.from_pretrained(MODEL)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {
                "type": "text",
                "text": PROMPT,
            },
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

torch.cuda.synchronize()
inference_start = time.perf_counter()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )
torch.cuda.synchronize()
inference_time_ms = (time.perf_counter() - inference_start) * 1000

generated = output[0, inputs["input_ids"].shape[1]:]
response = processor.decode(
    generated,
    skip_special_tokens=True,
)
detections = parse_detections(response)
annotated_image = draw_detections(image, detections)
if not cv2.imwrite(OUTPUT_PATH, annotated_image):
    raise OSError(f"Could not write annotated image to {OUTPUT_PATH}")

console = Console()
console.print(Panel(Markdown(PROMPT.strip()), title="Prompt", border_style="green"))
console.print(Panel(Markdown(response), title="Qwen response", border_style="cyan"))
console.print(f"[bold]Detected red levers:[/] [yellow]{len(detections)}[/]")
console.print(f"[bold]Inference time:[/] [yellow]{inference_time_ms:.2f} ms[/]")
console.print(f"[bold]Annotated image:[/] [yellow]{OUTPUT_PATH}[/]")

if SHOW_PREVIEW:
    window_name = "Qwen detections"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
