#!/bin/bash
# Submit a full supervision sweep (every config in configs/) for one or more
# datasets as slurm jobs, organised under a single named experiment folder.
#
# Usage:
#   ./submit_experiments.sh [options] <experiment-name> <dataset> [<dataset> ...]
#
# Each <dataset> is one of:
#   - a leaf data-root (a directory with train/ and test/ directly inside it)
#   - name=path, to control the subfolder name for a leaf data-root
#   - a parent directory: if it has no train/ of its own, every immediate
#     subdirectory that DOES have a train/ is auto-discovered and expanded
#     into its own dataset, named after that subfolder
#
# Auto-discovery (pass the parent once, get every rnd-2-cp*/rnd-3/... below it):
#   ./submit_experiments.sh baseline ../data/BowTie-NONSME-Groundtruth
#
# Equivalent explicit form:
#   ./submit_experiments.sh baseline \
#       cp1=../data/BowTie-NONSME-Groundtruth/rnd-2-cp1 \
#       cp2=../data/BowTie-NONSME-Groundtruth/rnd-2-cp2 \
#       cp3=../data/BowTie-NONSME-Groundtruth/rnd-2-cp3 \
#       rnd3=../data/BowTie-NONSME-Groundtruth/rnd-3
#
# The two forms can be mixed, and multiple parents can be passed at once.
#
# Produces:
#   results-baseline/
#       cp1/
#           unsup/ weakly/ fully/ mixed_r25/ mixed_r50/ mixed_r75/   (one per config)
#           aggregated/<supervision>/last.csv
#       cp2/ ...
#       rnd3/ ...
#
# Options:
#   --nodes n1,n2     Nodes to round-robin datasets across (default: waccamaw01,waccamaw02)
#   --configs a,b     Config files to run (default: all configs/custom_*.txt)
#   --results-root D  Parent dir for results-<experiment-name> (default: .)
#   --dry-run         Print the sbatch commands without submitting
#
# Every dataset is pinned to one node (all its supervision jobs run there);
# datasets are distributed round-robin over --nodes.

set -euo pipefail

NODES="waccamaw01,waccamaw02"
CONFIGS=""
RESULTS_ROOT="."
DRY_RUN=0

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nodes)        NODES="$2"; shift 2 ;;
        --configs)      CONFIGS="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage ;;
        --*)            echo "ERROR: unknown option: $1" >&2; usage 1 ;;
        *)              break ;;
    esac
done

[[ $# -ge 2 ]] || { echo "ERROR: need an experiment name and at least one dataset" >&2; usage 1; }

EXPERIMENT="$1"; shift
RAW_DATASETS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- resolve datasets: expand parent dirs into their leaf subdirectories ----
# A directory is a "leaf" dataset root if it has train/ directly inside it
# (the Custom datamodule's expected layout). Anything else is treated as a
# parent: every immediate child with its own train/ becomes one dataset,
# named after that child's folder name.
resolve_dataset() {
    local entry="$1" name path
    if [[ "$entry" == *=* ]]; then
        name="${entry%%=*}"
        path="${entry#*=}"
    else
        path="$entry"
        name=""
    fi
    path="${path%/}"
    [[ -d "$path" ]] || { echo "ERROR: dataset path not found: $path" >&2; exit 1; }

    if [[ -d "$path/train" ]]; then
        [[ -n "$name" ]] || name="$(basename "$path")"
        echo "$name=$path"
        return
    fi

    local child found=0
    for child in "$path"/*/; do
        [[ -d "$child" ]] || continue
        child="${child%/}"
        if [[ -d "$child/train" ]]; then
            echo "$(basename "$child")=$child"
            found=1
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo "ERROR: '$path' has no train/ subdirectory and none of its" >&2
        echo "  immediate children do either -- not a valid dataset root or parent." >&2
        exit 1
    fi
}

DATASETS=()
for entry in "${RAW_DATASETS[@]}"; do
    # NOTE: must use command substitution, not `< <(...)` process substitution --
    # the latter runs resolve_dataset in a background subshell whose `exit 1`
    # would NOT stop this script under `set -e`.
    resolved_output="$(resolve_dataset "$entry")"
    while IFS= read -r resolved; do
        [[ -n "$resolved" ]] && DATASETS+=("$resolved")
    done <<< "$resolved_output"
done

# ---- resolve configs ----
if [[ -n "$CONFIGS" ]]; then
    IFS=',' read -r -a CONFIG_FILES <<< "$CONFIGS"
else
    CONFIG_FILES=("$SCRIPT_DIR"/configs/custom_*.txt)
fi
for cfg in "${CONFIG_FILES[@]}"; do
    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; exit 1; }
done

# ---- resolve nodes ----
IFS=',' read -r -a NODE_LIST <<< "$NODES"

if [[ $DRY_RUN -eq 0 ]] && ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on this machine. Use --dry-run to preview commands." >&2
    exit 1
fi

# supervision tier name from config filename:
#   custom_unsup.txt -> unsup, custom_weakly_sup.txt -> weakly,
#   custom_mixed_sup_r25.txt -> mixed_r25, custom_fully_sup.txt -> fully
sup_name() {
    local n
    n="$(basename "$1" .txt)"
    n="${n#custom_}"
    n="${n/_sup/}"
    echo "$n"
}

RESULTS_BASE="$RESULTS_ROOT/results-$EXPERIMENT"
echo "Experiment: $EXPERIMENT"
echo "Results under: $RESULTS_BASE/"
echo ""

n_jobs=0
for i in "${!DATASETS[@]}"; do
    entry="${DATASETS[$i]}"
    if [[ "$entry" == *=* ]]; then
        ds_name="${entry%%=*}"
        data_root="${entry#*=}"
    else
        data_root="$entry"
        ds_name="$(basename "${data_root%/}")"
    fi
    [[ -d "$data_root" ]] || { echo "ERROR: data root not found: $data_root" >&2; exit 1; }

    node="${NODE_LIST[$(( i % ${#NODE_LIST[@]} ))]}"
    results_dir="$RESULTS_BASE/$ds_name"
    echo "── $ds_name  ($data_root)  →  node $node"

    for cfg in "${CONFIG_FILES[@]}"; do
        sup="$(sup_name "$cfg")"
        cmd=(sbatch --nodelist="$node" "$SCRIPT_DIR/run_slurm_train.sh"
             --config "$cfg"
             --data-root "$data_root"
             --results-dir "$results_dir"
             --run-name "$sup")
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "  [dry-run] ${cmd[*]}"
        else
            out="$("${cmd[@]}")"
            echo "  $sup: $out"
        fi
        n_jobs=$((n_jobs + 1))
    done
    echo ""
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: $n_jobs jobs would be submitted."
else
    echo "Submitted $n_jobs jobs. Monitor with: squeue -u \$USER"
fi
