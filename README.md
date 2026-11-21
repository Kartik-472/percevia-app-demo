# Perceviya
One of the two demo products made for the hackathon

## Track

Accessibility and Assistive Technology

## Problem

Existing visual assistance apps often rely on cloud connectivity, incur recurring per-user compute or subscription costs, introduce network latency during real-time movement, and stream camera footage off-device. Visually impaired users in low-connectivity environments or using budget Android hardware need reliable, low-latency, private obstacle awareness without constant internet access.

## Solution

Perceiviya is an offline, privacy-first Android application for real-time proximity and obstacle warnings on budget (4GB RAM) devices. It runs Ultralytics YOLO11n (`320x320` ONNX) for object detection, an optional MiDaS Small (`256x256` ONNX) depth model with a 2D bounding-box fallback for relative proximity (`URGENT`, `NEAR`, `FAR`), spatial zone mapping (`left`, `ahead`, `right`), haptic alerts, cooldown-throttled Text-to-Speech, and tap-to-describe scene summaries—all on-device with zero network calls.

## Tech Stack

* **Language & Platform:** Kotlin, Android (`AppCompatActivity`)
* **Camera & UI:** CameraX (`PreviewView`, `ImageAnalysis`), custom bounding-box overlay
* **Inference:** ONNX Runtime Android (`yolo11n.onnx`, optional `depth.onnx`)
* **Feedback:** Android `TextToSpeech`, `Vibrator` / `VibratorManager`

## Project Setup

1. Export `yolo11n.onnx` (`imgsz=320`) and optionally `depth.onnx` (`256x256` MiDaS Small) and place them in `app/src/main/assets/`.
2. Ensure `package com.example.perceiviya` matches your namespace and `AndroidManifest.xml` activity declaration.
3. Build and install on an Android device with camera permissions enabled.

## Competitive Edge & Scalability

* **Zero marginal server cost:** All inference and TTS run locally, allowing broad distribution without cloud inference bills scaling with users.
* **Offline & low-latency:** Operates without Wi-Fi or mobile data in dead zones, transit, or indoor spaces.
* **Privacy by default:** Camera frames never leave the device.

## Limitations & Next Steps

* Distance is relative (normalized depth / box heuristics), not metric, and depends on camera framing and lighting.
* Next steps: calibrate proximity thresholds against false warnings, optimize frame-skipping for battery/FPS on 4GB RAM devices, and validate with blind and low-vision testers.
