#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-ubfc}"
if [[ "$dataset" == "ubfc" ]]; then
  raw="data/raw/UBFC-rPPG/DATASET_2"
  cache="data/cache/UBFC-rPPG-128"
  standard="configs/ubfc_rppgvit.yaml"
  light="configs/ubfc_light_rppgvit.yaml"
  size=128
elif [[ "$dataset" == "mmpd" ]]; then
  raw="data/raw/MMPD-mini"
  cache="data/cache/MMPD-mini-80"
  standard="configs/mmpd_rppgvit.yaml"
  light="configs/mmpd_light_rppgvit.yaml"
  size=80
else
  echo "dataset must be ubfc or mmpd" >&2
  exit 2
fi

.venv/bin/python scripts/check_data_ready.py "$dataset" --root "$raw" --deep \
  --output "outputs/data_checks/${dataset}.json"
.venv/bin/python scripts/preprocess.py "$dataset" --input "$raw" --output "$cache" --size "$size"
.venv/bin/python scripts/train.py --config "$standard" --resume
standard_output="$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$standard'))['experiment']['output_dir'])")"
.venv/bin/python scripts/evaluate.py --config "$standard" --checkpoint "$standard_output/best.pt"
.venv/bin/python scripts/train.py --config "$light" --resume
light_output="$(.venv/bin/python -c "import yaml; print(yaml.safe_load(open('$light'))['experiment']['output_dir'])")"
.venv/bin/python scripts/evaluate.py --config "$light" --checkpoint "$light_output/best.pt"
.venv/bin/python scripts/benchmark.py --model all --frames 128 --size 128 \
  --count-ops --output "outputs/benchmarks/${dataset}_local.json"
.venv/bin/python scripts/compare_to_paper.py --benchmark "outputs/benchmarks/${dataset}_local.json"
