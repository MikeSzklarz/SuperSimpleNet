#!/bin/bash
# Submit a resolution sweep, restricted to whichever supervision tiers you ask
# for, as slurm jobs -- standalone (submits sbatch jobs directly, the same way
# submit_experiments.sh does; does not shell out to it).
#
# Usage:
#   ./submit_sweep.sh --experiment-name <name> --data-root <dataset> [<dataset> ...] \
#       [--unsup] [--weakly] [--mixed [ratios]] [--fully] \
#       (--square | --rect | --sizes HxW[,HxW...]) \
#       [--nodes n1,n2] [--results-root D] [--dry-run] [--batch-override HxW=N ...]
#
# --data-root accepts the same leaf/parent/name=path forms as
# submit_experiments.sh (auto-discovery of rnd-2-cp1/rnd-3/... included).
#
# Supervision flags -- at least one required, each maps to an existing config,
# no new config files needed:
#   --unsup              configs/custom_unsup.txt
#   --weakly             configs/custom_weakly_sup.txt
#   --mixed [25,50,75]   configs/custom_mixed_sup_r{ratio}.txt -- all three
#                        ratios if no value given, or restrict e.g. --mixed 50
#   --fully              configs/custom_fully_sup.txt
#
# Resolution ladders -- exactly one of these three required:
#   --square   256x256 384x384 512x512 768x768    (batch 32/16/8/4)
#   --rect     256x256 384x512 576x768 768x1024   (batch 32/16/8/4)
#   --sizes    explicit comma list of HxW pairs, e.g. --sizes 256x256,640x640
#              (any size not in the built-in batch table needs
#              --batch-override HxW=N or the script errors out)
#
# Example -- a new dataset that can only train unsupervised/weakly, square ladder:
#   ./submit_sweep.sh --experiment-name rnd-4-zoomed-in \
#       --data-root ../data/rnd-4-zoomed-in \
#       --unsup --weakly --square
#
# Produces (same convention as submit_experiments.sh):
#   results-rnd-4-zoomed-in/
#       <dataset>/
#           unsup_256x256/  weakly_256x256/
#           unsup_384x384/  weakly_384x384/
#           unsup_512x512/  weakly_512x512/
#           unsup_768x768/  weakly_768x768/
#           aggregated/<sup>_<H>x<W>/last.csv
#
# Each job's slurm name (visible in squeue): <experiment-name>_<dataset>_<sup>_<H>x<W>
#
# Options:
#   --experiment-name N   Required. Results are written to results-N/
#   --data-root D [D...]  Required. One or more datasets; may repeat
#   --unsup/--weakly/--fully  Include that supervision tier (any combination)
#   --mixed [ratios]      Include mixed supervision; bare = all of 25,50,75
#   --square/--rect/--sizes  Resolution ladder (exactly one required)
#   --batch-override HxW=N  Override/define the batch size for one size; may repeat
#   --nodes n1,n2          Nodes to round-robin datasets across (default: waccamaw01,waccamaw02)
#   --results-root D       Parent dir for results-<experiment-name> (default: .)
#   --dry-run              Print the sbatch commands without submitting
#
# Every individual job (each dataset x size x supervision combination) is
# round-robined over --nodes -- unlike submit_experiments.sh, which pins a
# whole dataset's jobs to one node. A resolution sweep is commonly run
# against a single dataset, so pinning per-dataset would leave every node but
# the first completely idle; round-robining per job spreads the sweep across
# all nodes given, single dataset or many.

set -euo pipefail

EXPERIMENT=""
RAW_DATASETS=()
NODES="waccamaw01,waccamaw02"
RESULTS_ROOT="."
DRY_RUN=0
LADDER=""
CUSTOM_SIZES=()

SUP_UNSUP=0
SUP_WEAKLY=0
SUP_MIXED=0
MIXED_RATIOS=""
SUP_FULLY=0

declare -A BATCH_OVERRIDES=()

usage() { sed -n '2,58p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name) EXPERIMENT="$2"; shift 2 ;;
        --data-root)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                RAW_DATASETS+=("$1")
                shift
            done
            ;;
        --unsup)  SUP_UNSUP=1; shift ;;
        --weakly) SUP_WEAKLY=1; shift ;;
        --fully)  SUP_FULLY=1; shift ;;
        --mixed)
            SUP_MIXED=1
            shift
            if [[ $# -gt 0 && "$1" != --* ]]; then
                MIXED_RATIOS="$1"
                shift
            fi
            ;;
        --square) LADDER="square"; shift ;;
        --rect)   LADDER="rect"; shift ;;
        --sizes)
            LADDER="custom"
            IFS=',' read -r -a CUSTOM_SIZES <<< "$2"
            shift 2
            ;;
        --batch-override)
            BATCH_OVERRIDES["${2%%=*}"]="${2#*=}"
            shift 2
            ;;
        --nodes)        NODES="$2"; shift 2 ;;
        --results-root) RESULTS_ROOT="$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1; shift ;;
        -h|--help)      usage ;;
        --*)            echo "ERROR: unknown option: $1" >&2; usage 1 ;;
        *)              echo "ERROR: unexpected argument '$1' (did you forget --data-root?)" >&2; usage 1 ;;
    esac
done

[[ -n "$EXPERIMENT" ]] || { echo "ERROR: --experiment-name is required" >&2; usage 1; }
[[ ${#RAW_DATASETS[@]} -ge 1 ]] || { echo "ERROR: --data-root is required (at least one dataset)" >&2; usage 1; }
[[ -n "$LADDER" ]] || { echo "ERROR: one of --square, --rect, --sizes is required" >&2; usage 1; }
(( SUP_UNSUP || SUP_WEAKLY || SUP_MIXED || SUP_FULLY )) || {
    echo "ERROR: at least one of --unsup/--weakly/--mixed/--fully is required" >&2
    usage 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- resolve datasets: expand parent dirs into their leaf subdirectories ----
# (identical logic to submit_experiments.sh, duplicated here since this script
# is standalone rather than shelling out to it)
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
    resolved_output="$(resolve_dataset "$entry")"
    while IFS= read -r resolved; do
        [[ -n "$resolved" ]] && DATASETS+=("$resolved")
    done <<< "$resolved_output"
done

# ---- resolve supervision configs ----
CONFIG_FILES=()
[[ $SUP_UNSUP -eq 1 ]]  && CONFIG_FILES+=("$SCRIPT_DIR/configs/custom_unsup.txt")
[[ $SUP_WEAKLY -eq 1 ]] && CONFIG_FILES+=("$SCRIPT_DIR/configs/custom_weakly_sup.txt")
[[ $SUP_FULLY -eq 1 ]]  && CONFIG_FILES+=("$SCRIPT_DIR/configs/custom_fully_sup.txt")
if [[ $SUP_MIXED -eq 1 ]]; then
    if [[ -n "$MIXED_RATIOS" ]]; then
        IFS=',' read -r -a ratios <<< "$MIXED_RATIOS"
    else
        ratios=(25 50 75)
    fi
    for r in "${ratios[@]}"; do
        CONFIG_FILES+=("$SCRIPT_DIR/configs/custom_mixed_sup_r${r}.txt")
    done
fi
for cfg in "${CONFIG_FILES[@]}"; do
    [[ -f "$cfg" ]] || { echo "ERROR: config not found: $cfg" >&2; exit 1; }
done

# ---- resolve sizes + batch table ----
declare -A BATCH_TABLE=(
    ["256x256"]=32
    ["384x384"]=16
    ["512x512"]=8
    ["768x768"]=4
    ["384x512"]=16
    ["576x768"]=8
    ["768x1024"]=4
)

case "$LADDER" in
    square) SIZES=("256x256" "384x384" "512x512" "768x768") ;;
    rect)   SIZES=("256x256" "384x512" "576x768" "768x1024") ;;
    custom) SIZES=("${CUSTOM_SIZES[@]}") ;;
esac
[[ ${#SIZES[@]} -ge 1 ]] || { echo "ERROR: --sizes given but no sizes parsed" >&2; exit 1; }

for size in "${SIZES[@]}"; do
    [[ "$size" == *x* ]] || { echo "ERROR: size '$size' must be in HxW form (e.g. 512x512)" >&2; exit 1; }
    if [[ -z "${BATCH_OVERRIDES[$size]+x}" && -z "${BATCH_TABLE[$size]+x}" ]]; then
        echo "ERROR: no batch size known for '$size'. Add --batch-override ${size}=N" >&2
        exit 1
    fi
done

# ---- resolve nodes ----
IFS=',' read -r -a NODE_LIST <<< "$NODES"

if [[ $DRY_RUN -eq 0 ]] && ! command -v sbatch >/dev/null; then
    echo "ERROR: sbatch not found on this machine. Use --dry-run to preview commands." >&2
    exit 1
fi

# supervision tier name from config filename (identical to submit_experiments.sh):
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
echo "Sizes:      ${SIZES[*]}"
echo "Configs:    $(for c in "${CONFIG_FILES[@]}"; do basename "$c"; done | tr '\n' ' ')"
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

    results_dir="$RESULTS_BASE/$ds_name"
    echo "── $ds_name  ($data_root)"

    for size in "${SIZES[@]}"; do
        h="${size%x*}"
        w="${size#*x}"
        batch="${BATCH_OVERRIDES[$size]:-${BATCH_TABLE[$size]:-}}"

        for cfg in "${CONFIG_FILES[@]}"; do
            # Round-robin every individual job (dataset x size x supervision)
            # over --nodes, not just once per dataset -- a resolution sweep is
            # commonly run against a single dataset, and pinning per-dataset
            # would leave every node but the first idle in that case.
            node="${NODE_LIST[$(( n_jobs % ${#NODE_LIST[@]} ))]}"
            sup="$(sup_name "$cfg")"
            run_name="${sup}_${h}x${w}"
            job_name="${EXPERIMENT}_${ds_name}_${run_name}"
            job_name="${job_name// /_}"
            cmd=(sbatch --nodelist="$node" --job-name="$job_name" "$SCRIPT_DIR/run_slurm_train.sh"
                 --config "$cfg"
                 --data-root "$data_root"
                 --results-dir "$results_dir"
                 --run-name "$run_name"
                 --image-size "$h" "$w"
                 --batch "$batch")
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "  [dry-run] (node $node) ${cmd[*]}"
            else
                out="$("${cmd[@]}")"
                echo "  $run_name (node $node): $out"
            fi
            n_jobs=$((n_jobs + 1))
        done
    done
    echo ""
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run: $n_jobs jobs would be submitted."
else
    echo "Submitted $n_jobs jobs. Monitor with: squeue -u \$USER"
fi
