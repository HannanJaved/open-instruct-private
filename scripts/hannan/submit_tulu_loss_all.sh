#!/usr/bin/env bash
# submit_tulu_loss_all.sh
# Generates SBATCH job scripts from tulu_loss.sh template for multiple models/steps/ranks
# Usage: ./submit_tulu_loss_all.sh [--dry-run] [--submit]

set -euo pipefail

BASE_DIR="/data/horse/ws/hama901h-BFTranslation"
TEMPLATE="$BASE_DIR/tulu_loss.sh"
OUTDIR="$BASE_DIR/generated_sbatch_jobs"

DRY_RUN=true
SUBMIT=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift;;
    --submit) SUBMIT=true; DRY_RUN=false; shift;;
    -h|--help)
      echo "Usage: $0 [--dry-run] [--submit]"
      echo "--dry-run  : only generate job scripts, don't sbatch them (default)"
      echo "--submit   : generate and sbatch each job"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

mkdir -p "$OUTDIR"

# Parameters
RANKS=(64 256 1024)
LRS=(1e5 1e6 5e5)
WRS=(001 005 010)
STEPS=(6000 12000 18000 24000 30000 36000 42000 48000 final)

# Read template once
if [[ ! -f "$TEMPLATE" ]]; then
  echo "Template $TEMPLATE not found" >&2
  exit 1
fi
TEMPLATE_CONTENT=$(<"$TEMPLATE")

# Helper to create job script content by replacing placeholders
create_job_script() {
  local rank=$1
  local lr=$2
  local wr=$3
  local step=$4
  local name="alpha_${lr}_${wr}"
  local step_suffix="$step"
  local step_dir=""
  if [[ "$step_suffix" == "final" ]]; then
    step_dir="final"
  else
    step_dir="step_${step_suffix}"
  fi
  local model_subpath="checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank${rank}/${name}/${step_dir}"
  local jobname="Loss_Rank${rank}/${name}/${step_suffix}"
  local out="logs/losses/Rank${rank}/${name}/${step_suffix}.out"
  local err="logs/losses/Rank${rank}/${name}/${step_suffix}.err"

  # Replace SBATCH header fields and the model path in the python command
  local content="$TEMPLATE_CONTENT"
  # Replace job name line
  content=$(echo "$content" | sed -E "s|^#SBATCH --job-name=.*$|#SBATCH --job-name=${jobname}|")
  content=$(echo "$content" | sed -E "s|^#SBATCH --output=.*$|#SBATCH --output=${out}|")
  content=$(echo "$content" | sed -E "s|^#SBATCH --error=.*$|#SBATCH --error=${err}|")

  # Replace the model path in the srun python command. We assume the template contains the exact path used currently.
  content=$(echo "$content" | sed -E "s|--model_name_or_path[[:space:]]+[^\\[:space:]]+|--model_name_or_path ${BASE_DIR}/${model_subpath}|")

  # Replace the output_dir argument so results are stored under eval_outputs/Rank{rank}/{lr}/{wr}/{step_dir}
  local eval_out_dir="${BASE_DIR}/eval_outputs/Rank${rank}/LR_${lr}/WR_${wr}/${step_dir}"
  content=$(echo "$content" | sed -E "s|--output_dir[[:space:]]+[^[:space:]]+|--output_dir ${eval_out_dir}|")

  echo "$content"
}

# Generate and optionally submit jobs
generated=0
for rank in "${RANKS[@]}"; do
  for lr in "${LRS[@]}"; do
    for wr in "${WRS[@]}"; do
      for step in "${STEPS[@]}"; do
        step_dir=""
        if [[ "$step" == "final" ]]; then
          step_dir="final"
        else
          step_dir="step_${step}"
        fi
        # Use 'final' as literal directory name for final
        jobfile="$OUTDIR/job_Rank${rank}_alpha_${lr}_${wr}_${step}.sh"
        create_job_script "$rank" "$lr" "$wr" "$step" > "$jobfile"
        chmod +x "$jobfile"
        echo "Generated $jobfile"
        generated=$((generated+1))
        if [[ "$SUBMIT" == true ]]; then
          # Check if eval_results.json already exists
          eval_results_path="${BASE_DIR}/tulu_val_loss/Rank${rank}/LR_${lr}/WR_${wr}/${step_dir}/eval_results.json"
          if [[ -f "$eval_results_path" ]]; then
            echo "Skipping submission of $jobfile as eval_results.json already exists at $eval_results_path"
            continue
          fi
          echo "Submitting $jobfile..."
          sbatch "$jobfile"
        fi
      done
    done
  done
done

echo "Generated $generated job scripts in $OUTDIR"
if [[ "$DRY_RUN" == true ]]; then
  echo "Dry-run mode: no jobs were submitted. Rerun with --submit to sbatch them."
fi
