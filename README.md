# NFL Play Outcome Classifier

A computer vision + machine learning pipeline that predicts the outcome of an NFL play (e.g. run, completed pass, incomplete pass, sack) directly from broadcast game film — no play-by-play text or tracking chips required as input, just video.

Built as a graduate Computer Vision team project at Case Western Reserve University.

## What it does

Given raw broadcast video of a play, the pipeline:

1. **Detects** players, the ball, and field landmarks (hashes, yard markers, pylons) frame-by-frame using YOLO.
2. **Identifies** player roles (QB, center, linebacker, defensive back, skill position) and jersey numbers from cropped detections.
3. **Maps** pixel-space detections into real-world field coordinates (yards) via homography, using the field markings as reference points.
4. **Tracks** players/ball across frames to build a spatiotemporal picture of the play as it develops.
5. **Engineers features** from that spatiotemporal data — detection geometry, time-since-snap, role-aware positional counts, and per-role speed — at the frame level.
6. **Classifies** the play outcome with a gradient-boosted (XGBoost) model, trained with class-balanced sample weights and calibrated probabilities, evaluated at both the frame level and the play level (via majority vote across a play's frames).

This combination of detection-derived spatiotemporal features improved model performance by ~70% over the initial baseline, and the final pipeline reports precision/recall/confusion matrices per class at both the frame and play level.

## Pipeline stages and where to find them

| Stage | Notebook(s) | Purpose |
|---|---|---|
| Label play outcomes | `Label_PlayOutcomes.ipynb` | Interactive labeling UI (ipywidgets) for tagging play videos with one of 13 fine-grained outcomes (e.g. `Run_TFL`, `Pass_COM_20+`, `Sack`), later collapsed into a 4-class target for modeling. Produces `play_labels.csv`. |
| Label jersey numbers | `Label_Number.ipynb` | Splits/labels cropped jersey-number images into buckets (10s/20s/30s/40s/50s) for the number classifier. |
| Player & field detection | `football_yolo_training.ipynb` | Trains a YOLO detector on the Roboflow "Football Player Detection" dataset (`data/Football Player Detection.v7i.yolov11`) to find `ball`, `player`, `ref`, `hash`, `five_hash`, `marker`, `pylon`, `number`. |
| Player role classification | `PlayerRollDetection.ipynb` | Trains/evaluates a YOLO classifier that labels detected players by position group (CENTER, DB, LB, QB, SKILL). |
| Jersey number classification | `TrainNumberClassifier.ipynb`, `number_classifier.yaml` | Trains a YOLO classification model on cropped jersey numbers to help disambiguate players. |
| Multi-model fusion + homography | `combine_two_yolo_models.ipynb` (and `_new`/`_updated` variants) | Runs the field-detection model and the player-detection model on the same frame, merges both sets of detections, and maps pixel coordinates to on-field yard coordinates via homography using the field markings. |
| Video-level tracking | `video_tracker.ipynb` | Runs detection + tracking across an entire play's video to follow players/ball frame-to-frame. |
| Feature engineering + outcome model | `prelim_play_classifier.ipynb` (early version), `finalmodel.ipynb` (final pipeline) | Builds play-level and frame-level features from tracking/helmet/video-metadata (in the style of the Kaggle "NFL Player Contact Detection" dataset) plus the custom play labels, trains the XGBoost classifier, and evaluates it. |

`PreliminaryWork.ipynb` and `Untitled.ipynb` are earlier scratch/exploration notebooks kept for reference.

## Model details (`finalmodel.ipynb`)

- **Target:** 4-class play outcome (`coarse_label`), derived from 13 fine-grained hand-labeled outcomes.
- **Features:** frame-level detection geometry (bounding-box counts, area/aspect-ratio statistics), `time_since_snap` (aligned via video metadata and frame rate), and role-aware features (per-team/position player counts and mean speed from tracking data).
- **Model:** `XGBClassifier` (`multi:softprob`) inside a scikit-learn `Pipeline` with median imputation, trained with inverse-frequency class weights to handle class imbalance, then wrapped in `CalibratedClassifierCV` (sigmoid) to produce better-calibrated probabilities.
- **Splitting:** `GroupShuffleSplit` grouped by `game_play`, so frames from the same play never leak across train/validation.
- **Evaluation:** frame-level classification report + confusion matrix, and a play-level confusion matrix/accuracy using majority vote across a play's sampled frames, plus a summary of prediction confidence to check for overconfidence.

## Repository layout

```
├── Label_PlayOutcomes.ipynb              # Labeling UI for play outcome classes
├── Label_Number.ipynb                    # Labeling/splitting for jersey number crops
├── football_yolo_training.ipynb          # Train YOLO player/field detector
├── PlayerRollDetection.ipynb             # Train/evaluate player position classifier
├── TrainNumberClassifier.ipynb           # Train jersey number classifier
├── combine_two_yolo_models*.ipynb        # Fuse field + player models, homography mapping
├── video_tracker.ipynb                   # Cross-frame player/ball tracking
├── prelim_play_classifier.ipynb          # Early feature engineering + XGBoost pass
├── finalmodel.ipynb                      # Final feature pipeline + XGBoost play outcome model
├── number_classifier.yaml                # Class config for jersey number classifier
├── data/
│   ├── Football Player Detection.v7i.yolov11/   # Roboflow detection dataset (train/valid/test)
│   ├── number_crops_labeled/                    # Jersey number crops by bucket (10/20/30/40/50/ignore)
│   ├── number_crops_split*/                     # Train/val/test splits of the above
├── runs/
│   ├── detect/                           # YOLO detection training/validation runs
│   └── classify/                         # YOLO number-classifier training runs
├── yolo11n.pt / yolo11s.pt / yolov8n-cls.pt      # Pretrained YOLO weights used as starting points
```

Note: the NFL Player Contact Detection tracking/helmet/video-metadata files referenced by `prelim_play_classifier.ipynb` and `finalmodel.ipynb` (`train_baseline_helmets.csv`, `train_player_tracking.csv`, `train_video_metadata.csv`) are not stored in this repo — they come from the corresponding Kaggle competition dataset and should be placed under `data/nfl-player-contact-detection/` before running those notebooks, alongside a `play_labels.csv` produced by `Label_PlayOutcomes.ipynb`.

## Tech stack

- **Detection/classification:** Ultralytics YOLO (v8/v11), PyTorch
- **Modeling:** XGBoost, scikit-learn (pipelines, calibration, evaluation)
- **Data handling:** pandas, NumPy
- **Vision/visualization:** OpenCV, Matplotlib, Seaborn
- **Labeling tooling:** ipywidgets, Jupyter

## Getting started

This is a notebook-driven research project rather than a packaged application — there's no `requirements.txt` yet, but the notebooks install what they need inline (`pip install ultralytics opencv-python xgboost scikit-learn seaborn pandas`).

1. Set up a Python environment with the packages above (a CUDA-capable GPU is recommended for YOLO training).
2. Point the dataset paths in each notebook's `CONFIG` section (e.g. `DATA_ROOT`, `DATA_DIR`, model weight paths) at your local data.
3. Run the notebooks roughly in the order listed in the pipeline table above: label data → train detection/classification models → fuse detections and map to field coordinates → build features → train/evaluate the outcome classifier in `finalmodel.ipynb`.

## Project status

This repo reflects an iterative research process — several notebooks (`combine_two_yolo_models.ipynb`, `_new`, `_updated`) are successive versions of the same experiment kept for reference rather than a single canonical script, and paths/configs are set up for the original authors' local environment. `finalmodel.ipynb` is the most complete end-to-end version of the outcome-classification stage.

## Acknowledgments

- Player/field detection dataset: ["Football Player Detection"](https://universe.roboflow.com/vayvay/football-player-detection-xdjq7) via Roboflow (CC BY 4.0).
- Tracking/helmet/video metadata schema based on the Kaggle NFL Player Contact Detection competition dataset.
