# Camera Intelligence System – Person Re-Identification

## Overview

This project is a camera intelligence system for person re-identification using a recorded video stream.

The system detects people, extracts Re-ID embeddings, compares them with a persistent gallery, and assigns one of four statuses:

- KNOWN
- NEW
- UNCERTAIN
- NO_PERSON

The system does not identify real-world identities. It uses internal gallery person IDs only.

## Pipeline

```text
Video
   ↓
Frame Sampling
   ↓
Person Detection
   ↓
Person Crop
   ↓
Re-ID Embedding
   ↓
Gallery Matching
   ↓
Decision
   ↓
SQLite Storage
   ↓
Annotated Output
```

## AI and ML Components

### Person Detection

YOLOv8n is used to detect people in video frames.

### Person Re-Identification

OSNet x0_25 is used to extract 512-dimensional person embeddings.

The model uses pretrained weights from the Market-1501 Re-ID model.

### Matching

Embeddings are L2-normalized and compared using cosine similarity.

### Decision

The system classifies a detected person as:

- KNOWN
- NEW
- UNCERTAIN

based on similarity thresholds.

## Data Storage

SQLite is used for persistent structured storage.

The database contains:

- `gallery`
- `sessions`
- `visit_events`

Gallery embeddings remain available after restarting the application.

## Project Structure

```text
camera-reid-capstone/
├── README.md
├── requirements.txt
├── config.py
├── data/
├── models/
├── src/
├── benchmark/
├── demo/
├── tests/
└── report/
```

## Requirements

- Python 3.11
- PyTorch
- TorchVision
- TorchReID
- Ultralytics
- OpenCV
- NumPy

The system is configured to run on CPU.

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

Place the required model files inside the `models/` directory.

## Running the Pipeline

Run the recorded-video pipeline with:

```powershell
python src\pipeline.py
```

Input video:

```text
demo/input_video.mp4
```

Annotated output:

```text
demo/output_annotated.mp4
```

## Evaluation

The project uses the Market-1501 dataset for quantitative person Re-ID evaluation.

The evaluation metrics are:

- Rank-1
- Rank-5
- mAP

Final measured results will be added after completing the benchmark.

## Threshold Calibration

The current decision thresholds are provisional and were not used as validated results.

Final threshold values will be calibrated using Market-1501 similarity results and documented after evaluation.

## Testing

The project includes tests for:

- Person detection
- Re-ID embeddings
- Similarity matching
- In-memory gallery
- SQLite persistence
- Sessions and visit events
- Market-1501 dataset validation
- Final pipeline execution

## Failure Handling

The system handles:

- Unreadable video
- Invalid frames
- No detected person
- Invalid person crops
- Embedding errors
- Detection errors
- Uncertain similarity

Expected failures are handled without crashing the entire pipeline.

## Current Limitations

- The demo uses a recorded video.
- CPU inference is used.
- The system selects the primary detected person in each frame.
- No real-world identity recognition is performed.
- Final benchmark results and calibrated thresholds will be added after evaluation.

## Privacy

The system uses internal person IDs instead of real names or real-world identities.

No real personal identities are assigned to detected people.