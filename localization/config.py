"""Yo'llar va retrieval/matching konstantalari."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MPLCONFIG_DIR = PROJECT_DIR / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

QUERY_IMAGE = PROJECT_DIR / "query" / "test.jpg"
POSE_PLOT = PROJECT_DIR / "colmap_pose_plot.png"
POSE_HTML = PROJECT_DIR / "colmap_pose_view.html"
CAMERA_CENTERS_TXT = PROJECT_DIR / "reference_camera_centers.txt"
MATCH_VIS_IMAGE = PROJECT_DIR / "query_best_match.png"
LOCALIZATION_SUMMARY_TXT = PROJECT_DIR / "localization_summary.txt"
WEB_RUNTIME_DIR = PROJECT_DIR / "web_runtime"
WEB_UPLOAD_DIR = WEB_RUNTIME_DIR / "uploads"
WEB_RESULTS_DIR = WEB_RUNTIME_DIR / "results"
WEB_QUERY_IMAGE = WEB_UPLOAD_DIR / "current_query.jpg"
WEB_POSE_PLOT = WEB_RESULTS_DIR / "current_pose_plot.png"
WEB_MATCH_VIS = WEB_RESULTS_DIR / "current_match_vis.png"
WEB_SUMMARY_TXT = WEB_RESULTS_DIR / "current_summary.txt"
DEFAULT_WORKSPACE_DIR = PROJECT_DIR / "colmap_workspace"
REF_IMAGE_DIR = PROJECT_DIR / "ref_images"
CACHE_DIR = PROJECT_DIR / ".cache"
REF_GLOBAL_CACHE = CACHE_DIR / "reference_global_descriptors.pkl"

# Web upload himoyasi: demo local bo'lsa ham, juda katta fayl serverni sekinlashtirib qo'ymasin.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

MATCH_VIS_MAX_LINES = 200
TOP_REFERENCE_RESULTS = 3

# Faqat query_best_match.png chizish: noto'g'ri mosliklarni kamaytirish
MATCH_VIZ_LOWE_RATIO = 0.68
MATCH_VIZ_LOWE_RATIO_RELAXED = 0.74
MATCH_VIZ_F_THRESH = 1.8
MATCH_VIZ_F_THRESH_RELAXED = 2.5
MATCH_VIZ_H_THRESH = 3.0
MATCH_VIZ_H_THRESH_RELAXED = 4.0
MATCH_VIZ_RANSAC_CONF = 0.999
MATCH_VIZ_USE_MUTUAL_NN = True

# SIFT / matching
SIFT_MAX_FEATURES_QUERY = 6000
SIFT_MAX_FEATURES_VIZ = 6000
SIFT_MAX_FEATURES_PAIR = 4000
LOWE_RATIO_DEFAULT = 0.78

# PnP/fallback: global o'xshashlik bo'yicha 200 tagacha candidate tekshiriladi.
PNP_GLOBAL_SHORTLIST = 200

# PnP natijasini tanlash: faqat inlier emas, global (4 qism) o'xshashlik ham (img_06 vs img_09 kabi xatolar kamayadi)
# rank = num_inliers + PNP_GLOBAL_RANK_WEIGHT * (global_sum / n_quads), n_quads ~ 4
PNP_GLOBAL_RANK_WEIGHT = 55.0
# PnP pose matched reference kameradan juda uzoqqa sakrasa, bu odatda noto'g'ri PnP.
PNP_REF_DISTANCE_MIN = 2.0
PNP_REF_DISTANCE_MEDIAN_MULTIPLIER = 4.0
# 6 ta inlier matematik minimum, lekin indoor corridor uchun juda zaif. Past bo'lsa fallback xavfsizroq.
PNP_MIN_ACCEPTED_INLIERS = 12
# Fallback: quad inlier yig'indisi + global; global baland bo'lsa ustunlik
FALLBACK_GLOBAL_RANK_WEIGHT = 50.0
FALLBACK_GOOD_MATCH_SCALE = 0.012

# Fallback full ranking: spatial balance va geometry sanity.
FALLBACK_MIN_QUAD_INLIERS = 20
FALLBACK_MIN_BALANCE = 0.15
FALLBACK_SCORE_FUSION_WEIGHT = 1.0
FALLBACK_SCORE_INLIER_WEIGHT = 0.5
FALLBACK_SCORE_MATCH_WEIGHT = 0.2
FALLBACK_SCORE_GLOBAL_WEIGHT = 50.0
FALLBACK_SCORE_BALANCE_WEIGHT = 200.0
FALLBACK_SCORE_DISPERSION_WEIGHT = 0.05
FALLBACK_SCORE_AREA_RATIO_WEIGHT = 100.0
FALLBACK_MIN_AREA_RATIO = 0.05
FALLBACK_MAX_AREA_RATIO = 4.0

# Top-1 va top-2 juda yaqin bo'lsa, localization "aniq" emas deb ko'rsatamiz.
AMBIGUOUS_RELATIVE_MARGIN = 0.06
MEDIUM_CONFIDENCE_RELATIVE_MARGIN = 0.15

# Query ni 2x2 = 4 qismga bo'lib: har qismdan o'xshashlik (yuz bo'ylab)
QUERY_RETRIEVAL_GRID_ROWS = 2
QUERY_RETRIEVAL_GRID_COLS = 2
QUERY_QUAD_MIN_SIDE_PX = 40
# PnP: query SIFT har kvadratdan teng nfeatures (chap/o'ng muvozanat)
QUERY_PNP_SIFT_PER_QUAD = 1500

# Fundamental / homografiya RANSAC
FUNDAMENTAL_RANSAC_THRESH = 3.0
HOMOGRAPHY_RANSAC_THRESH = 5.0

MODEL_DIR_ENV = os.environ.get("COLMAP_MODEL_DIR", str(DEFAULT_WORKSPACE_DIR / "sparse" / "0"))
DATABASE_PATH_ENV = os.environ.get("COLMAP_DATABASE_PATH", str(DEFAULT_WORKSPACE_DIR / "database.db"))
