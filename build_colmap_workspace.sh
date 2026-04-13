#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="$PROJECT_DIR/ref_images"
WORKSPACE_DIR="$PROJECT_DIR/colmap_workspace"
BACKUP_ROOT="$PROJECT_DIR/colmap_backups"
DATABASE_PATH="$WORKSPACE_DIR/database.db"
SPARSE_DIR="$WORKSPACE_DIR/sparse"
MODEL_DIR="$SPARSE_DIR/0"
LOG_DIR="$WORKSPACE_DIR/logs"
IMAGE_LIST_PATH="$WORKSPACE_DIR/image_list.txt"
RESET_MODE="${1:-}"

if [[ "$RESET_MODE" == "--reset" && -d "$WORKSPACE_DIR" ]]; then
  mkdir -p "$BACKUP_ROOT"
  TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
  BACKUP_DIR="$BACKUP_ROOT/colmap_workspace_$TIMESTAMP"
  mv "$WORKSPACE_DIR" "$BACKUP_DIR"
  echo "Eski workspace backup qilindi: $BACKUP_DIR"
fi

mkdir -p "$WORKSPACE_DIR" "$SPARSE_DIR" "$LOG_DIR"

if [[ ! -d "$IMAGE_DIR" ]]; then
  echo "ref_images papkasi topilmadi: $IMAGE_DIR"
  exit 1
fi

if [[ -f "$DATABASE_PATH" ]]; then
  echo "Database allaqachon bor: $DATABASE_PATH"
  echo "Qayta qurish uchun avval project ichidagi colmap_workspace/database.db ni o'chiring yoki nomini o'zgartiring."
  exit 1
fi

find "$IMAGE_DIR" -maxdepth 1 \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  -exec basename {} \; | sort -V > "$IMAGE_LIST_PATH"

echo "1/3 Feature extraction..."
colmap feature_extractor \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_DIR" \
  --image_list_path "$IMAGE_LIST_PATH" \
  --ImageReader.single_camera 1 \
  --FeatureExtraction.num_threads 4 \
  --FeatureExtraction.use_gpu 0 \
  --FeatureExtraction.max_image_size 2400 \
  --SiftExtraction.max_num_features 12000 \
  2>&1 | tee "$LOG_DIR/feature_extractor.log"

echo "2/3 Exhaustive matching..."
colmap exhaustive_matcher \
  --database_path "$DATABASE_PATH" \
  --FeatureMatching.use_gpu 0 \
  --FeatureMatching.guided_matching 1 \
  --TwoViewGeometry.min_num_inliers 12 \
  --TwoViewGeometry.max_error 6 \
  2>&1 | tee "$LOG_DIR/exhaustive_matcher.log"

echo "3/3 Sparse mapping..."
colmap mapper \
  --database_path "$DATABASE_PATH" \
  --image_path "$IMAGE_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.multiple_models 0 \
  --Mapper.min_model_size 5 \
  --Mapper.min_num_matches 10 \
  --Mapper.init_min_num_inliers 20 \
  --Mapper.abs_pose_min_num_inliers 12 \
  --Mapper.init_min_tri_angle 4 \
  --Mapper.filter_min_tri_angle 0.5 \
  2>&1 | tee "$LOG_DIR/mapper.log"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Sparse model yaralmadi. Loglarni tekshiring: $LOG_DIR"
  exit 1
fi

echo ""
echo "Tayyor."
echo "Workspace: $WORKSPACE_DIR"
echo "Database:  $DATABASE_PATH"
echo "Model:     $MODEL_DIR"
