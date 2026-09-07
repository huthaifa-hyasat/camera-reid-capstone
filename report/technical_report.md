انسخ هذا النص كاملًا والصقه داخل `report/technical_report.md`:

````markdown
# Technical Report

## 1. Project Overview

Camera Intelligence System – Person Re-Identification is a computer vision system that processes recorded video, detects people, extracts person re-identification features, compares them with a persistent gallery, and assigns a recognition status.

The system uses internal person IDs and does not identify real-world identities.

## 2. System Pipeline

```text
Video Input
    ↓
Frame Sampling
    ↓
Person Detection
    ↓
Person Crop
    ↓
Re-ID Embedding
    ↓
Cosine Similarity Matching
    ↓
Decision
    ↓
SQLite Storage
    ↓
Annotated Video Output
````

## 3. AI and ML Components

### 3.1 Person Detection

YOLOv8n is used to detect people in each processed video frame.

Only detections belonging to the person class are used by the system.

### 3.2 Person Re-Identification

OSNet x0_25 is used to extract a 512-dimensional embedding from each detected person crop.

The model uses pretrained Market-1501 weights.

Each embedding is L2-normalized before matching.

### 3.3 Similarity Matching

The query embedding is compared with the stored gallery embeddings using cosine similarity.

The highest similarity score is used by the decision component.

### 3.4 Decision States

The system produces one of the following states:

* KNOWN
* NEW
* UNCERTAIN
* NO_PERSON

The decision is based on calibrated similarity thresholds.

## 4. Data and Benchmark

Market-1501 is used for quantitative evaluation of the person re-identification component.

The dataset used by the project contains:

* 3,368 query images
* 19,732 gallery images

The benchmark follows the Market-1501 evaluation setup and excludes junk images and same-camera matches where required.

## 5. Evaluation Results

The Re-ID model was evaluated on the Market-1501 dataset.

The measured results are:

| Metric | Result |
| ------ | -----: |
| Rank-1 | 91.78% |
| Rank-5 | 97.21% |
| mAP    | 78.06% |

These results measure the person re-identification component on the Market-1501 benchmark.

They do not represent end-to-end accuracy on the recorded demonstration video.

## 6. Threshold Calibration

Similarity thresholds were calibrated using Market-1501 similarity scores.

The calibration produced:

| Parameter        |  Value |
| ---------------- | -----: |
| EER Threshold    | 0.5310 |
| EER              | 2.656% |
| Match Threshold  | 0.5800 |
| Reject Threshold | 0.5310 |

The calibration used:

* 49,649 genuine pairs
* 44,108,556 impostor pairs

The calibrated values were then used by the final video pipeline.

## 7. Persistent Storage

SQLite is used for persistent structured storage.

The database contains three main tables:

### gallery

Stores:

* Internal person ID
* Re-ID embedding
* First-seen timestamp
* Optional label

### sessions

Stores:

* Session ID
* Session start time
* Session end time

### visit_events

Stores:

* Event ID
* Session ID
* Person ID
* Timestamp
* Recognition status
* Similarity score

This allows the system to preserve gallery information and event history across application restarts.

## 8. Final Pipeline Test

The final pipeline was tested using the recorded demonstration video.

The test processed:

* 838 frames read
* 838 frames processed

The resulting states were:

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

## 9. Failure Handling

The system includes handling for expected failure cases.

These include:

* Unreadable video source
* Invalid video dimensions
* Invalid frames
* No detected person
* Invalid person crops
* Detection errors
* Embedding errors
* Empty gallery
* Invalid embeddings
* Uncertain similarity results

Expected failures are handled without crashing the entire pipeline.

## 10. Testing

The project includes tests for the main system components:

* Person detection
* Re-ID embedding extraction
* Cosine similarity matching
* In-memory gallery
* SQLite persistence
* Sessions and visit events
* Market-1501 dataset structure
* Model checkpoint compatibility
* Final pipeline execution

The tests were used throughout development to verify individual components before integrating the complete pipeline.

## 11. Development Approach

The system was developed incrementally.

The main components were implemented and tested separately before integration.

The development process included:

1. Person detection
2. Re-ID embedding extraction
3. Similarity matching
4. Gallery management
5. SQLite persistence
6. Decision logic
7. Video pipeline integration
8. Market-1501 benchmark
9. Threshold calibration
10. Final pipeline testing

This approach helped identify and validate errors at the component level before the final integration.

## 12. Limitations

The current system has several limitations:

* The primary demonstration uses a recorded video.
* CPU inference is used.
* The system selects a primary detected person from each frame.
* The system does not perform real-world identity recognition.
* The system does not train or fine-tune a model.
* Performance depends on the quality of person detection and video input.

## 13. Privacy

The system does not assign real names or real-world identities to detected people.

It uses internal gallery person IDs for re-identification.

No facial identity recognition is performed.

## 14. Reproducibility

The project includes the source code, requirements, model configuration, benchmark scripts, tests, and documented evaluation results required to reproduce the main experiments.

The Market-1501 dataset is kept outside the Git repository and is used locally for benchmarking.

Generated files such as database files, embeddings, and large video outputs are excluded from the Git repository.

```
