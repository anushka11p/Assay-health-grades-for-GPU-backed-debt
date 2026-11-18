#!/usr/bin/env bash
# Builds the census site from census/raw, refusing to publish anything unverified.
#
#   scripts/build.sh            # writes census/census.json and site/
#
# 1. builds the assay agent
# 2. verifies the signature of EVERY report in census/raw (stops on the first failure)
# 3. checks the transparency log's signed tree head, if there is a log
# 4. grades the GPUs and generates the static site
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> building assay"
(cd agent && go build -o ../.assay-bin ./cmd/assay 2>/dev/null)
trap 'rm -f .assay-bin' EXIT

echo "==> verifying reports"
count=0
for f in census/raw/*.json; do
  [[ "$f" == *.proof.json ]] && continue
  pub="${f%.json}.pub"
  if [[ -f "$pub" ]]; then
    ./.assay-bin verify --pubkey "$pub" "$f" > /dev/null || { ./.assay-bin verify --pubkey "$pub" "$f"; exit 1; }
  else
    ./.assay-bin verify "$f" > /dev/null || { ./.assay-bin verify "$f"; exit 1; }
  fi
  count=$((count + 1))
done
echo "    $count reports verified"

LOG_ARGS=()
if [[ -f log/head.json ]]; then
  echo "==> checking transparency log"
  ./.assay-bin log head --dir log
  LOG_ARGS=(--log ../log)
fi

echo "==> grading and building site"
(cd grading && python3 -m assay_grading --raw ../census/raw ${LOG_ARGS[@]+"${LOG_ARGS[@]}"} --census ../census/census.json --site ../site)
