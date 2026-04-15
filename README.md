# Indoor Localization MVP

Indoor corridor-style image localization prototype built around stable structural cues instead of color-heavy matching.

This project is designed for practical indoor re-localization where wall paint, lighting, furniture, and temporary clutter may change over time, but the corridor geometry usually stays similar:

- ceiling / wall boundaries
- pillar edges
- door-frame structure
- corridor direction lines
- upper-wall layout

The current system uses a line-first retrieval stage and a stronger second-stage verification step so that "more lines" does not automatically mean "better match".

## What It Does

Given a query image, the system:

1. extracts structural line features from a selected ROI
2. finds the most similar reference frames by line-based structure
3. verifies shortlisted candidates with structural consistency + local feature checks
4. selects the best reference / station
5. shows the result on a logical map
6. optionally tries PnP when usable COLMAP data exists

Primary goal:

- robust reference / station localization

Secondary goal:

- optional true pose upgrade via PnP

If PnP fails, the system falls back cleanly to reference-level localization.

## Core Idea

This is **not** a full Cupix-style 3D localization system.

It is a working indoor localization prototype focused on:

- simplicity
- debug visibility
- stable fallback behavior
- structure over color

## Current Pipeline

### Stage 1: Structural line shortlist

For each reference image:

- convert to grayscale
- apply ROI masking
- ignore most of the floor region
- detect edges with Canny
- detect lines with HoughLinesP
- compute structural descriptors:
  - orientation similarity
  - parallel-line consistency
  - spatial distribution
  - zone-based structure histograms
  - perspective consistency

This stage returns a top-K shortlist.

### Stage 2: Verification

For shortlisted candidates only:

- SIFT matching
- fundamental matrix geometric verification
- region consistency
- structural similarity
- spatial coverage / balance checks

Final top-1 is selected from this second stage, not from line count alone.

### Optional Stage 3: PnP

If compatible COLMAP image / 3D correspondences exist:

- solvePnPRansac is attempted

If not accepted:

- result mode stays `reference_fallback`

## Key Features

- structure-first localization
- floor suppression in line ranking
- ROI-aware line extraction
- candidate score breakdowns for debugging
- HTML inspection view
- upload-based local test server
- ARKit dataset preparation helper

## Project Layout

```text
.
├── features.py             # line features, SIFT, matching helpers
├── io_colmap.py            # COLMAP + coords + dataset metadata loading
├── localization.py         # main localization pipeline
├── retrieval.py            # shortlist and final ranking logic
├── visualize.py            # summary, PNG, HTML outputs
├── run.py                  # CLI localization entry point
├── run_html.py             # simple local upload UI
├── server.py               # richer upload-based local web app
├── prepare_arkit_refs.py   # build clean reference subsets from ARKit captures
├── ref_images/             # reference datasets
├── query/                  # query images
└── colmap_workspace/       # optional COLMAP workspace
```

## Setup

Create and activate a virtual environment, then install the dependencies you need.

Typical dependencies used by this project:

- Python 3.10+
- OpenCV
- NumPy
- Matplotlib
- Flask

If you already have a working `.venv`, you can keep using it.

## Quick Start

### 1. Run with default query

```bash
.venv/bin/python run.py
```

If `ref_images/` itself does not contain images, the pipeline automatically tries to pick a usable reference subfolder.

### 2. Run with a specific reference set

```bash
.venv/bin/python run.py \
  --ref-dir "ref_images/dataset-20260410-142733-2_prepared" \
  --query "query/test.jpg" \
  --no-pnp
```

### 3. Run with the old corridor set

```bash
.venv/bin/python run.py \
  --ref-dir "ref_images/corridor" \
  --coords "coords.txt"
```

## Web Usage

### Simple local upload UI

```bash
.venv/bin/python run_html.py --serve
```

### Rich upload server

```bash
.venv/bin/python server.py \
  --ref-dir "ref_images/dataset-20260410-142733-2_prepared" \
  --no-pnp
```

Then open:

```text
http://127.0.0.1:5000
```

The web view includes:

- image upload
- line visualization
- interactive location map
- top candidates
- hover tooltip with scores
- double-click image modal on map points

## Preparing ARKit References

If you have an ARKit capture folder with many frames, prepare a lighter reference set:

```bash
.venv/bin/python prepare_arkit_refs.py \
  --dataset-dir "ref_images/dataset-20260410-142733 2" \
  --output-dir "ref_images/dataset-20260410-142733-2_prepared" \
  --stride 3
```

This creates:

- a cleaned reference folder
- selected JPG frames
- generated `coords.txt`
- a small preparation summary

## Outputs

Running the pipeline produces:

- `localization_summary.txt`
- `query_line_viz.png`
- `query_best_match.png`
- `pose_plot.png`
- `pnp_inliers.png`
- `colmap_pose_view.html`

### What the summary includes

- final selected frame
- top candidates
- line shortlist
- ROI debug
- ignored floor fraction
- score breakdown
- rejection reasons for similar but wrong candidates

## Current Scoring Logic

### First-stage shortlist

Line shortlist is based on structure, not raw line count:

- line direction similarity
- parallel-line consistency
- spatial distribution similarity
- zone similarity
- perspective consistency

### Final ranking

Final ranking uses a hybrid of:

- structural similarity
- point verification
- region consistency
- coverage
- balance

So even if a wrong frame has many lines, it should lose when the structure and verification are weaker.

## Recommended Workflow

For best results:

1. prepare a clean reference folder
2. keep query images outside the reference set
3. use `--no-pnp` unless COLMAP matches the same reference images
4. inspect `localization_summary.txt`
5. inspect `query_line_viz.png` and `query_best_match.png`
6. use the HTML page to compare top candidates visually

## Limitations

This is still an MVP, so some things are intentionally scoped:

- best for corridor-like indoor scenes
- not a full general indoor SLAM system
- highly repetitive environments can still produce lookalikes
- PnP only works when compatible COLMAP correspondences exist
- dataset quality and reference coverage still matter a lot

## Good Next Steps

- cluster similar reference frames into station groups
- add stronger lookalike suppression between neighboring ceiling patterns
- improve README examples for your exact dataset
- add unit tests for scoring and candidate rejection
- add a small sample dataset config file

## Branch / Development Notes

Recent work on this repo focused on:

- structural line-based shortlist retrieval
- reduced floor influence
- stronger second-stage verification
- richer debug output
- local HTML upload inspection

If you are reviewing changes from the pushed branch, inspect:

- `features.py`
- `localization.py`
- `retrieval.py`
- `visualize.py`
- `server.py`

## License / Ownership

Add your preferred license here if you want to make the repository public in a cleaner way.
