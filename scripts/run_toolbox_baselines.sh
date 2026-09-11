#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dataset="${1:-ubfc}"
requested_model="${2:-all}"
if [[ "$dataset" == "ubfc" ]]; then
  raw="$root/data/raw/UBFC-rPPG/DATASET_2"
elif [[ "$dataset" == "mmpd-mini" ]]; then
  raw="$root/data/raw/MMPD-mini"
else
  echo "dataset must be ubfc or mmpd-mini" >&2
  exit 2
fi

"$root/.venv/bin/python" "$root/scripts/check_data_ready.py" "${dataset%%-*}" --root "$raw" --deep \
  --output "$root/outputs/data_checks/${dataset}.json"
"$root/.venv/bin/python" "$root/scripts/generate_toolbox_baseline_configs.py"

for model in tscan physnet deepphys physformer efficientphys; do
  if [[ "$requested_model" != "all" && "$requested_model" != "$model" ]]; then
    continue
  fi
  case "$model" in
    tscan) display="TSCAN" ;;
    physnet) display="PhysNet" ;;
    deepphys) display="DeepPhys" ;;
    physformer) display="PhysFormer" ;;
    efficientphys) display="EfficientPhys" ;;
  esac
  config="$root/configs/toolbox_baselines/${dataset}/${model}.yaml"
  if [[ "$dataset" == "ubfc" && "$model" == "tscan" && -f "$root/data/cache/rppg-toolbox/ubfc/tscan/DataFileLists/UBFC-rPPG_TSCAN_within_622_0.8_1.0.csv" ]]; then
    config="$root/configs/toolbox_baselines/ubfc/tscan_cpu_cached.yaml"
  elif [[ "$dataset" == "ubfc" && "$model" == "deepphys" && -f "$root/data/cache/rppg-toolbox/ubfc/tscan/DataFileLists/UBFC-rPPG_TSCAN_within_622_0.8_1.0.csv" ]]; then
    config="$root/configs/toolbox_baselines/ubfc/deepphys_cpu_shared_cache.yaml"
  fi
  metrics="$root/outputs/toolbox-results/${dataset}/${model}/metrics.json"
  log_dir="$root/outputs/toolbox-results/${dataset}/${model}"
  run_log="$log_dir/training.log"
  if [[ -f "$metrics" && "${FORCE:-0}" != "1" ]]; then
    echo "Skipping completed ${dataset}/${display}: ${metrics}"
    continue
  fi
  mkdir -p "$log_dir"
  (
    cd "$root/third_party/rPPG-Toolbox"
    MPLCONFIGDIR="$root/.cache/matplotlib" \
      "$root/.venv/bin/python" main.py --config_file "$config" 2>&1 | tee "$run_log"
  )
  prediction="$(find "$root/outputs/toolbox/${dataset}/${display}" -path "*/saved_test_outputs/*_outputs.pickle" -type f | sort | tail -1)"
  if [[ -z "$prediction" ]]; then
    echo "No saved prediction pickle found for ${dataset}/${model}" >&2
    exit 3
  fi
  "$root/.venv/bin/python" "$root/scripts/collect_toolbox_metrics.py" \
    --input "$prediction" \
    --output "$metrics"
done

"$root/.venv/bin/python" "$root/scripts/compare_to_paper.py"
