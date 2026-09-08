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
````

## AI and ML Components

### Person Detection

YOLOv8n is used to detect people in video frames.

Only detections belonging to the person class are used by the system.

### Person Re-Identification

OSNet x0_25 is used to extract 512-dimensional person embeddings.

The model uses pretrained weights trained for the Market-1501 Re-ID task.

Each embedding is L2-normalized before matching.

### Matching

Embeddings are compared using cosine similarity.

The highest similarity score is used by the decision component.

### Decision

The system classifies a detected person as:

* KNOWN
* NEW
* UNCERTAIN

based on calibrated similarity thresholds.

## Data Storage

SQLite is used for persistent structured storage.

The database contains:

* `gallery`
* `sessions`
* `visit_events`

The `gallery` table stores person embeddings and internal person IDs.

The `sessions` table stores session start and end times.

The `visit_events` table stores recognition events, timestamps, statuses, and similarity scores.

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

* Python 3.11
* PyTorch
* TorchVision
* TorchReID
* Ultralytics
* OpenCV
* NumPy

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

The required models are:

* `yolov8n.pt`
* `osnet_x0_25_market1501.pth`

The Market-1501 dataset is used locally for benchmarking and is not included in the Git repository.

## Running the Pipeline

Run the recorded-video pipeline with:

```powershell
python src\pipeline.py
```

Input video:

```text
demo/input_video.mp4
```

The pipeline generates an annotated output video locally:

```text
demo/output_annotated.mp4
```

Generated video files are excluded from the Git repository because of their large size.

## Evaluation

The project uses the Market-1501 dataset for quantitative person Re-ID evaluation.

The dataset contains:

* 3,368 query images
* 19,732 gallery images

The evaluation metrics are:

* Rank-1
* Rank-5
* mAP

Final results:

| Metric | Result |
| ------ | -----: |
| Rank-1 | 91.78% |
| Rank-5 | 97.21% |
| mAP    | 78.06% |

These results measure the Re-ID component on the Market-1501 benchmark and do not represent end-to-end accuracy on the demonstration video.

## Threshold Calibration

Similarity thresholds were calibrated using Market-1501 similarity scores.

The final thresholds are:

| Parameter        |  Value |
| ---------------- | -----: |
| EER Threshold    | 0.5310 |
| EER              | 2.656% |
| Match Threshold  | 0.5800 |
| Reject Threshold | 0.5310 |

The calibration used:

* 49,649 genuine pairs
* 44,108,556 impostor pairs

The calibrated thresholds are used by the final video pipeline.

## Testing

The project includes tests for:

* Person detection
* Re-ID embeddings
* Similarity matching
* In-memory gallery
* SQLite persistence
* Sessions and visit events
* Market-1501 dataset validation
* Model checkpoint compatibility
* Final pipeline execution

The Market-1501 dataset validation test completed successfully.

```text
1 passed, 1 warning
```

The warning is related to Cython acceleration being unavailable in TorchReID. The evaluation continued successfully using the Python implementation.

## Final Pipeline Test

The final pipeline was tested using the recorded demonstration video.

The pipeline processed:

* 838 frames read
* 838 frames processed

Results:

| Status       | Count |
| ------------ | ----: |
| KNOWN        |   479 |
| UNCERTAIN    |   350 |
| NEW          |     0 |
| NO_PERSON    |     9 |
| EMBED_ERROR  |     0 |
| DETECT_ERROR |     0 |

The final pipeline completed successfully and generated an annotated output video.

These counts describe the behavior of the demonstration pipeline and are not a ground-truth accuracy measurement.

## Failure Handling

The system handles:

* Unreadable video
* Invalid video dimensions
* Invalid frames
* No detected person
* Invalid person crops
* Embedding errors
* Detection errors
* Empty gallery
* Invalid embeddings
* Uncertain similarity results

Expected failure conditions are handled without crashing the entire pipeline.

## Experiments and Iterative Improvements

The project was developed and improved through several testing stages.

### Experiment 1: Initial Re-ID Model

The first pipeline used pretrained OSNet weights and cosine similarity matching.

The model produced valid 512-dimensional embeddings and the complete video pipeline was successfully executed.

### Experiment 2: Market-1501 Pretrained Weights

The Re-ID model was updated to use OSNet x0_25 weights trained for the Market-1501 Re-ID task instead of generic ImageNet weights.

The embeddings were then evaluated quantitatively using the Market-1501 benchmark.

The resulting performance was:

* Rank-1: 91.78%
* Rank-5: 97.21%
* mAP: 78.06%

### Experiment 3: Threshold Calibration

The initial thresholds were provisional.

Market-1501 similarity scores were used to calibrate the decision thresholds.

The final thresholds became:

* Match threshold: 0.5800
* Reject threshold: 0.5310

These calibrated values were then used in the final video pipeline.

### Experiment 4: Final Pipeline Validation

The final pipeline was executed again after threshold calibration and integration changes.

The pipeline successfully processed all 838 frames without detection or embedding errors.

The final test confirmed that the complete pipeline operated successfully from video input through detection, embedding, matching, decision, storage, and annotated output.

## Current Limitations

* The demonstration uses a recorded video.
* CPU inference is used.
* The system selects the primary detected person in each frame.
* No real-world identity recognition is performed.
* The system does not train or fine-tune a model.
* Performance depends on person detection and video quality.
* Additional independent testing under different people, lighting, distances, and backgrounds was not completed.

## Privacy

The system uses internal person IDs instead of real names or real-world identities.

No real personal identities are assigned to detected people.

No facial identity recognition is performed.

## Reproducibility

The repository contains:

* Source code
* Requirements file
* Configuration
* Database schema
* Benchmark scripts
* Tests
* Evaluation results
* Threshold calibration results
* Technical report
* Testing evidence
* README setup and run instructions

The Market-1501 dataset is kept outside the Git repository and is used locally for benchmarking.

Generated files such as database files, embedding caches, and large video outputs are excluded from the Git repository.