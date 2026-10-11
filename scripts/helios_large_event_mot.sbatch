#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
#SBATCH --signal=B:USR1@600
#
# Slurm job body for one stage of the large event MOT experiment.
#
# Submit only through the runner, which checks dependencies and the GPU-hour
# budget, records the reservation, and supplies account, partition, resources,
# wall time, and LARGE_EVENT_MOT_VENV:
#   python -m src.experiments.large_event_mot submit --runs-root RUNS --stage STAGE
#
# Usage (as called by sbatch): helios_large_event_mot.sbatch RUNS_ROOT STAGE
#
# A login shell (bash -l) provides the module command. The environment
# follows the node architecture: GH200 nodes (aarch64) load
# ML-bundle/25.10, CPU nodes (x86_64) load GCCcore/13.2.0 and Python/3.11.5;
# then the virtual environment in LARGE_EVENT_MOT_VENV is activated. The shell
# is replaced by the runner (exec), so SIGUSR1 from --signal reaches it
# directly; it forwards the signal to the running child, which saves and
# pauses. The job never resubmits itself.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 RUNS_ROOT STAGE" >&2
    exit 2
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "refusing to run outside a Slurm job; submit through the runner" >&2
    exit 2
fi
if [[ -z "${LARGE_EVENT_MOT_VENV:-}" || ! -f "${LARGE_EVENT_MOT_VENV}/bin/activate" ]]; then
    echo "LARGE_EVENT_MOT_VENV does not point to a virtual environment" >&2
    exit 2
fi

runs_root="$1"
stage="$2"
umask 077

case "$(uname -m)" in
    aarch64) modules=(ML-bundle/25.10) ;;
    x86_64) modules=(GCCcore/13.2.0 Python/3.11.5) ;;
    *) echo "unsupported architecture $(uname -m)" >&2; exit 2 ;;
esac
if ! command -v module >/dev/null 2>&1; then
    echo "environment modules unavailable; expected ${modules[*]}" >&2
    exit 2
fi
# Module and activation scripts may read unset variables.
set +u
module add "${modules[@]}"
# shellcheck disable=SC1091
source "${LARGE_EVENT_MOT_VENV}/bin/activate"
set -u

echo "job ${SLURM_JOB_ID} stage ${stage} on $(hostname -s) ($(uname -m)), $(date --iso-8601=seconds)"
export PYTHONUNBUFFERED=1
exec python -m src.experiments.large_event_mot job --runs-root "${runs_root}" --stage "${stage}"
