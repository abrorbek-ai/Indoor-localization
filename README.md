# Indoor Localization MVP

<p align="left">
  <img src="https://img.shields.io/badge/status-MVP-orange" alt="status">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/opencv-line%20%2B%20sift-success" alt="opencv">
  <img src="https://img.shields.io/badge/localization-structure--first-purple" alt="structure first">
</p>

Indoor corridor-style image localization prototype built around stable structural cues instead of color-heavy matching.

This project is designed for practical indoor re-localization where wall paint, lighting, furniture, and temporary clutter may change over time, but the corridor geometry usually stays similar:

- ceiling / wall boundaries
- pillar edges
- door-frame structure
- corridor direction lines
- upper-wall layout

The current system uses a line-first retrieval stage and a stronger second-stage verification step so that "more lines" does not automatically mean "better match".

## Highlights

- structure-first retrieval instead of color-heavy matching
- floor suppression to avoid brick-pattern bias
- line shortlist plus second-stage structural verification
- optional PnP fallback path
- local web UI for upload-based inspection
- debug-friendly outputs for why a wrong candidate was rejected

## TL;DR

If you just want to run it:

```bash
.venv/bin/python run.py \
  --ref-dir "ref_images/dataset-20260410-142733-2_prepared" \
  --query "query/test.jpg" \
  --no-pnp
```

If you want the upload UI:

```bash
.venv/bin/python server.py \
  --ref-dir "ref_images/dataset-20260410-142733-2_prepared" \
  --no-pnp
```

Then open `http://127.0.0.1:5000`.

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

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy matplotlib flask
```

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

## Typical Workflow

1. prepare a reference folder
2. put the test image into `query/` or pass `--query`
3. run CLI or web mode
4. inspect summary + visual outputs
5. compare top candidates in the HTML report

## Demo Outputs

After a successful run, these are the most useful files:

| File | Purpose |
| --- | --- |
| `localization_summary.txt` | full ranking, shortlist, score breakdown, rejection reasons |
| `query_line_viz.png` | query ROI + detected structural lines + top shortlist references |
| `query_best_match.png` | verified local feature matches between query and final top-1 |
| `pose_plot.png` | logical station / map view |
| `colmap_pose_view.html` | interactive report with map and candidate inspection |

Recommended order:

1. open `localization_summary.txt`
2. open `query_line_viz.png`
3. open `query_best_match.png`
4. open `colmap_pose_view.html`

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

### Web page sections

The richer HTML / server view is organized into:

1. line features visualization
2. interactive location map
3. top candidates

This makes it easier to answer:

- what was shortlisted first
- why the final frame won
- which similar frames were rejected

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

### Debug details included now

- selected ROI height
- ignored floor fraction
- filtered line count
- first-stage line shortlist
- structural similarity
- final weighted score breakdown
- why similar wrong candidates were rejected

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

## Repository Notes

This repo currently focuses on a practical indoor localization prototype rather than a polished product package.

That means:

- the code is organized for iteration and debugging
- generated outputs are intentionally kept out of Git
- datasets are not bundled into the repository
- the main value is in the pipeline logic and inspection tools

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

## Suggested Public Repo Polish

If you want this repository to feel cleaner to outside visitors, the next easy upgrades are:

- add 2-3 screenshots into a `docs/` folder
- add a tiny sample reference set
- add a short demo GIF of the web page
- add a license file
- add an example `requirements.txt`

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
