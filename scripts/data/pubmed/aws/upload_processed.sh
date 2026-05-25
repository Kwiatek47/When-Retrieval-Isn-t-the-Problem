#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:?Set BUCKET first}"
: "${DATASET_NAME:=pubmed_reviews_v1}"

aws s3 sync \
  "data/processed/${DATASET_NAME}/" \
  "s3://${BUCKET}/processed/${DATASET_NAME}/"

aws s3 ls "s3://${BUCKET}/processed/${DATASET_NAME}/"

