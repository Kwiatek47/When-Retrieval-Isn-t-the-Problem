#!/usr/bin/env bash
set -euo pipefail

: "${BUCKET:?Set BUCKET first}"
: "${AWS_REGION:=eu-central-1}"

aws s3 mb "s3://${BUCKET}" --region "${AWS_REGION}"

aws s3api put-public-access-block \
  --bucket "${BUCKET}" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws s3api put-bucket-versioning \
  --bucket "${BUCKET}" \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket "${BUCKET}" \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3 ls "s3://${BUCKET}"

