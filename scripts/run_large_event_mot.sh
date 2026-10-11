#!/bin/bash -l
#
# User-facing entry point for the large event MOT pipeline on Helios.
#
# Every path is derived from one runtime input, LEMOT_ROOT, so the commands in
# docs/large_event_mot.md are exact and nothing account- or user-specific is
# stored in the repository. Nothing runs by default: every call names exactly
# one command, and stages are submitted one at a time through the runner
# (python -m src.experiments.large_event_mot), which checks dependencies, the
# frozen stage specification, and the GPU-hour budget.
#
# Usage: scripts/run_large_event_mot.sh COMMAND [ARGS...]
#
# Inspection (no side effects):
#   paths                      print the derived layout and runtime inputs
#
# External code and data (fetch-dsec and fetch-gen4 only inside a plgrid job):
#   fetch-code [--dry-run]     RVT and TrackEval checkouts at the pinned commits,
#                              RVT-B 1Mpx checkpoint (size and SHA-256 verified)
#   fetch-dsec [--dry-run]     DSEC-Detection labels, left events of every DSEC
#                              sequence listed in configs/large_event_mot/data.json,
#                              DSEC-MOT image timestamps; resumable curl, size and
#                              ZIP checks, then idempotent extraction
#   fetch-gen4 SPLIT [--delete-archive] [--dry-run]
#                              1Mpx train or val tar from the UZH mirror; resumable
#                              curl, published CRC32 check, extraction
#   layout-dsec-mot [--dry-run]
#                              DSEC-MOT root: annotations from
#                              LEMOT_DSEC_MOT_ANNOTATIONS plus image timestamps
#
# Environments (inside a job on the matching partition):
#   venv gh200|cpu [--dry-run] build or update a virtual environment
#   preflight-cpu              CPU-only preflight of the CPU environment
#
# Runner (login node; uses the CPU environment):
#   init [--dry-run]           write the immutable experiment manifest once
#   plan | status              dry-run report / stage status and budget
#   submit --stage S [...]     submit exactly one job for one stage
#   reconcile [...] | charge [...]
#                              ledger maintenance; arguments are passed through
#
# Runtime inputs (environment variables):
#   LEMOT_ROOT                 required; base directory (author-only, mode 700)
#   LEMOT_GPU_ACCOUNT          required by init; Slurm account <grant>-gpu-gh200
#   LEMOT_CPU_ACCOUNT          required by init; Slurm account <grant>-cpu
#   LEMOT_DSEC_MOT_ANNOTATIONS required by layout-dsec-mot; directory with
#                              train/<seq>.txt and test/<seq>.txt
#   LEMOT_SEED                 optional, default 0 (init)
#   LEMOT_PARALLEL             optional, parallel downloads, default 4
#   LEMOT_PYTHON               optional interpreter for runner commands instead of
#                              the CPU environment (local checks only)
#
# Exit status: 0 success, 1 failure, 2 usage or input error; runner commands
# return the runner's own exit codes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_CONFIG="${REPO_ROOT}/configs/large_event_mot/data.json"
REQUIREMENTS="${REPO_ROOT}/requirements/large_event_mot.txt"

RVT_URL="https://github.com/uzh-rpg/RVT.git"
RVT_COMMIT="b80f5683a6e2d5de65d4bde8105d796ccb50dbb1"
TRACKEVAL_URL="https://github.com/JonathonLuiten/TrackEval.git"
TRACKEVAL_COMMIT="12c8791b303e0a0b50f753af204249e622d0281a"
RVT_CKPT_URL="https://download.ifi.uzh.ch/rpg/RVT/checkpoints/1mpx/rvt-b.ckpt"
RVT_CKPT_BYTES="297169732"
RVT_CKPT_SHA256="65d83e68236e99290757c1305e06da6c0cc650c435eed03cea70e9ffed8a1572"
GEN4_URL="https://download.ifi.uzh.ch/rpg/RVT/datasets/gen4_tar"
DSEC_URL="https://download.ifi.uzh.ch/rpg/DSEC"
# Published by RVT for the UZH mirror tars (zlib CRC32) and their sizes.
declare -A GEN4_CRC32=([train]=d677488a [val]=72f13c3e)
declare -A GEN4_BYTES=([train]=1042041978880 [val]=201110487040)
# The CPU environment needs einops to build the model for probe and test exports;
# requirements/large_event_mot.txt pins it for aarch64 only.
CPU_EXTRA_PINS=(einops==0.8.1)
GH200_MODULES=(ML-bundle/25.10)
CPU_MODULES=(GCCcore/13.2.0 Python/3.11.5)

die() {
    echo "error: $*" >&2
    exit 2
}

usage() {
    sed -n '3,52p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

require_var() {
    local name
    for name in "$@"; do
        [[ -n "${!name:-}" ]] || die "${name} is not set (see: $0 help)"
    done
}

require_job() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "$1 must run inside a Slurm job on the $2 partition${3:+ ($3)}"
}

require_arch() {
    [[ "$(uname -m)" == "$1" ]] || die "$2 must run on an $1 node, this is $(uname -m)"
}

set_paths() {
    require_var LEMOT_ROOT
    ROOT="$(realpath -m "${LEMOT_ROOT}")"
    RAW="${ROOT}/raw"
    GEN4_ROOT="${RAW}/gen4"
    GEN4_TAR_DIR="${RAW}/gen4_tar"
    DSEC_ROOT="${RAW}/dsec"
    DSEC_DETECTION_DIR="${RAW}/dsec_detection"
    DSEC_DETECTION_LABELS="${DSEC_DETECTION_DIR}/labels"
    DSEC_MOT_ROOT="${RAW}/dsec_mot"
    EXTERNAL="${ROOT}/external"
    RVT_ROOT="${EXTERNAL}/RVT"
    TRACKEVAL_ROOT="${EXTERNAL}/TrackEval"
    RVT_CHECKPOINT="${EXTERNAL}/rvt-b.ckpt"
    VENV_ROOT="${ROOT}/venvs"
    MANIFEST_ROOT="${ROOT}/manifests"
    CACHE_ROOT="${ROOT}/cache"
    RUNS_ROOT="${ROOT}/runs"
    RESULTS_ROOT="${ROOT}/results"
}

load_modules() {
    if ! command -v module >/dev/null 2>&1; then
        die "environment modules unavailable; run this script directly (login shell), not through sh"
    fi
    # Module and activation scripts may read unset variables.
    set +u
    module add "$@"
    set -u
}

activate() {
    [[ -f "$1/bin/activate" ]] || die "virtual environment $1 missing (see: $0 venv)"
    set +u
    # shellcheck disable=SC1091
    source "$1/bin/activate"
    set -u
}

file_size() {
    stat -c %s "$1" 2>/dev/null || echo 0
}

remote_size() {
    # Runs in xargs children without pipefail, so the curl status is checked first.
    local headers
    headers="$(curl -sSIL --fail --connect-timeout 30 "$1")" || return 1
    awk 'tolower($1) == "content-length:" { size = $2 } END { gsub(/\r/, "", size); if (size == "") exit 1; print size }' <<< "${headers}"
}

# Download URL to OUT with resume, size check, and a ZIP test; the final name
# appears only after every check passed, so an existing file is complete.
fetch_one() {
    local url="$1" out="$2" expected="${3:-}" part="$2.part" size have attempt
    if [[ -f "${out}" ]]; then
        echo "complete  ${out}"
        return 0
    fi
    size="${expected}"
    if [[ -z "${size}" ]]; then
        size="$(remote_size "${url}")" || { echo "FAILED    ${url}: no size from the server" >&2; return 1; }
    fi
    mkdir -p "$(dirname "${out}")"
    for attempt in 1 2 3 4 5; do
        have="$(file_size "${part}")"
        if (( have > size )); then
            rm -f "${part}"
            have=0
        fi
        if (( have < size )); then
            curl -sS --fail -L -C - --connect-timeout 30 --speed-limit 102400 --speed-time 120 \
                -o "${part}" "${url}" || { sleep $((attempt * 20)); continue; }
        fi
        have="$(file_size "${part}")"
        if (( have != size )); then
            sleep $((attempt * 20))
            continue
        fi
        if [[ "${out}" == *.zip ]] && ! unzip -tq "${part}" >/dev/null 2>&1; then
            echo "corrupt   ${part}: ZIP test failed, downloading again" >&2
            rm -f "${part}"
            continue
        fi
        mv "${part}" "${out}"
        echo "done      ${out} (${size} B)"
        return 0
    done
    echo "FAILED    ${url} -> ${out} (have $(file_size "${part}") of ${size} B)" >&2
    return 1
}

fetch_list() {
    # stdin: lines "URL<TAB>OUT"; runs LEMOT_PARALLEL downloads at a time.
    local parallel="${LEMOT_PARALLEL:-4}"
    export -f fetch_one file_size remote_size
    # shellcheck disable=SC2016  # expanded by the child shell
    xargs -d '\n' -n 1 -P "${parallel}" bash -c 'fetch_one "${1%%$'"'"'\t'"'"'*}" "${1#*$'"'"'\t'"'"'}"' _
}

# --------------------------------------------------------------------------- code

git_checkout() {
    local url="$1" commit="$2" dir="$3"
    if [[ -d "${dir}/.git" ]]; then
        local head
        head="$(git -C "${dir}" rev-parse HEAD 2>/dev/null || true)"
        if [[ "${head}" == "${commit}" ]]; then
            [[ -z "$(git -C "${dir}" status --porcelain --untracked-files=no)" ]] ||
                die "${dir} is at ${commit} but has modified tracked files"
            echo "complete  ${dir} at ${commit}"
            return 0
        fi
        die "${dir} is at ${head:-no commit}, expected ${commit}; move it away and fetch again"
    fi
    [[ ! -e "${dir}" ]] || die "${dir} exists and is not a git checkout"
    git init -q "${dir}"
    git -C "${dir}" fetch -q --depth 1 "${url}" "${commit}"
    git -C "${dir}" checkout -q FETCH_HEAD
    echo "done      ${dir} at $(git -C "${dir}" rev-parse HEAD)"
}

cmd_fetch_code() {
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1 && shift
    [[ $# -eq 0 ]] || die "usage: fetch-code [--dry-run]"
    set_paths
    if (( dry )); then
        printf '%s @ %s -> %s\n' "${RVT_URL}" "${RVT_COMMIT}" "${RVT_ROOT}" \
            "${TRACKEVAL_URL}" "${TRACKEVAL_COMMIT}" "${TRACKEVAL_ROOT}"
        printf '%s -> %s (%s B, SHA-256 %s)\n' "${RVT_CKPT_URL}" "${RVT_CHECKPOINT}" \
            "${RVT_CKPT_BYTES}" "${RVT_CKPT_SHA256}"
        return 0
    fi
    mkdir -p "${EXTERNAL}"
    git_checkout "${RVT_URL}" "${RVT_COMMIT}" "${RVT_ROOT}"
    git_checkout "${TRACKEVAL_URL}" "${TRACKEVAL_COMMIT}" "${TRACKEVAL_ROOT}"
    fetch_one "${RVT_CKPT_URL}" "${RVT_CHECKPOINT}" "${RVT_CKPT_BYTES}" || exit 1
    local digest
    digest="$(sha256sum "${RVT_CHECKPOINT}" | cut -d' ' -f1)"
    if [[ "${digest}" != "${RVT_CKPT_SHA256}" ]]; then
        mv "${RVT_CHECKPOINT}" "${RVT_CHECKPOINT}.bad"
        echo "FAILED    ${RVT_CHECKPOINT}: SHA-256 ${digest}, moved to .bad" >&2
        exit 1
    fi
    echo "verified  ${RVT_CHECKPOINT} SHA-256 ${digest}"
}

# --------------------------------------------------------------------------- DSEC

# Sequence lists come from data.json, so only the sequences the pipeline uses are
# fetched. The DSEC download folder is "test" for the 12 original DSEC test
# sequences and "train" otherwise; zurich_city_16_a to 21_a and thun_02_a exist
# only in the DSEC-Detection archives of new sequences.
DSEC_PY='
import io, json, os, sys, zipfile

TEST = {"interlaken_00_a", "interlaken_00_b", "interlaken_01_a", "thun_01_a", "thun_01_b",
        "zurich_city_12_a", "zurich_city_13_a", "zurich_city_13_b", "zurich_city_14_a",
        "zurich_city_14_b", "zurich_city_14_c", "zurich_city_15_a"}
NEW = {"train": {"zurich_city_%d_a" % i for i in range(16, 22)}, "test": {"thun_02_a"}}
LABELS = "dsec-det_left_object_detections.zip"


def sequences(config):
    with open(config) as handle:
        data = json.load(handle)
    mot = {s for names in data["dsec_mot"]["splits"].values() for s in names}
    det = {s for names in data["dsec_detection"]["splits"].values() for s in names}
    return mot, det


def folder(seq):
    for split, names in NEW.items():
        if seq in names:
            return None
    return "test" if seq in TEST else "train"


def urls(config, base, raw):
    mot, det = sequences(config)
    rows = [(base + "/detection/" + LABELS, os.path.join(raw, "dsec_detection", "labels", LABELS))]
    for seq in sorted(mot | det):
        split = folder(seq)
        if split is None:
            continue
        stem = "%s/%s/%s" % (split, seq, seq)
        rows.append(("%s/%s_events_left.zip" % (base, stem),
                     os.path.join(raw, "dsec", "%s_events_left.zip" % stem)))
        if seq in mot:
            rows.append(("%s/%s_image_timestamps.txt" % (base, stem),
                         os.path.join(raw, "dsec", "%s_image_timestamps.txt" % stem)))
    for split, names in sorted(NEW.items()):
        if names & det:
            name = "%s_events.zip" % split
            rows.append(("%s/%s_object_detection_coarse/%s" % (base, split, name),
                         os.path.join(raw, "dsec_detection", "new_sequences", name)))
    for url, out in rows:
        print("%s\t%s" % (url, out))


def put(archive, info, target):
    """Extract one member atomically; a file of the right size is kept."""
    if os.path.isfile(target) and os.path.getsize(target) == info.file_size:
        return 0
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    with archive.open(info) as source, open(partial, "wb") as sink:
        while True:
            block = source.read(1 << 24)
            if not block:
                break
            sink.write(block)
    os.replace(partial, target)
    return 1


def extract(config, raw):
    mot, det = sequences(config)
    dsec = os.path.join(raw, "dsec")
    written = 0
    for seq in sorted(mot | det):
        split = folder(seq)
        if split is None:
            continue
        seq_dir = os.path.join(dsec, split, seq)
        with zipfile.ZipFile(os.path.join(seq_dir, "%s_events_left.zip" % seq)) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    target = os.path.join(seq_dir, "events_left", os.path.basename(info.filename))
                    written += put(archive, info, target)
    for split, names in sorted(NEW.items()):
        wanted = names & det
        if not wanted:
            continue
        path = os.path.join(raw, "dsec_detection", "new_sequences", "%s_events.zip" % split)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                parts = info.filename.split("/")
                if (info.is_dir() or len(parts) != 5 or parts[0] != split
                        or parts[1] not in wanted or parts[2:4] != ["events", "left"]):
                    continue
                target = os.path.join(dsec, split, parts[1], "events_left", parts[4])
                written += put(archive, info, target)
    labels_dir = os.path.join(raw, "dsec_detection", "labels")
    with zipfile.ZipFile(os.path.join(labels_dir, LABELS)) as outer:
        for name in outer.namelist():
            if not name.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                for info in inner.infolist():
                    parts = info.filename.split("/")
                    if (info.filename.endswith("tracks.npy") and "right" not in parts
                            and len(parts) >= 3 and parts[0] in ("train", "test")):
                        target = os.path.join(labels_dir, parts[0], parts[1], "tracks.npy")
                        written += put(inner, info, target)
    for seq in sorted(mot | det):
        split = folder(seq) or next(s for s, names in NEW.items() if seq in names)
        for name in ("events.h5", "rectify_map.h5"):
            path = os.path.join(dsec, split, seq, "events_left", name)
            if not os.path.isfile(path):
                sys.exit("missing after extraction: " + path)
    print("extracted %d new files; events of %d sequences present" % (written, len(mot | det)))


if sys.argv[1] == "urls":
    urls(*sys.argv[2:])
else:
    extract(*sys.argv[2:])
'

cmd_fetch_dsec() {
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1 && shift
    [[ $# -eq 0 ]] || die "usage: fetch-dsec [--dry-run]"
    set_paths
    local list
    list="$(python3 -I -c "${DSEC_PY}" urls "${DATA_CONFIG}" "${DSEC_URL}" "${RAW}")"
    if (( dry )); then
        printf '%s\n' "${list}"
        echo "dry run: $(wc -l <<< "${list}") files; nothing downloaded"
        return 0
    fi
    require_job fetch-dsec plgrid "use --dry-run to inspect"
    umask 077
    mkdir -p "${RAW}"
    printf '%s\n' "${list}" | fetch_list || { echo "some downloads failed; run fetch-dsec again to resume" >&2; exit 1; }
    python3 -I -c "${DSEC_PY}" extract "${DATA_CONFIG}" "${RAW}"
}

# --------------------------------------------------------------------------- 1Mpx

CRC_PY='
import sys, zlib
crc = 0
with open(sys.argv[1], "rb") as handle:
    while True:
        block = handle.read(1 << 24)
        if not block:
            break
        crc = zlib.crc32(block, crc)
print("%08x" % (crc & 0xFFFFFFFF))
'

cmd_fetch_gen4() {
    local split="" dry=0 delete=0 arg
    for arg in "$@"; do
        case "${arg}" in
            train|val) split="${arg}" ;;
            --dry-run) dry=1 ;;
            --delete-archive) delete=1 ;;
            *) die "usage: fetch-gen4 train|val [--delete-archive] [--dry-run]" ;;
        esac
    done
    [[ -n "${split}" ]] || die "usage: fetch-gen4 train|val [--delete-archive] [--dry-run]"
    set_paths
    local url="${GEN4_URL}/${split}.tar" tar="${GEN4_TAR_DIR}/${split}.tar"
    local marker="${GEN4_ROOT}/.${split}.extracted"
    if (( dry )); then
        echo "${url} -> ${tar} (${GEN4_BYTES[${split}]} B, CRC32 ${GEN4_CRC32[${split}]})"
        echo "extract -> ${GEN4_ROOT}/${split}/ (marker ${marker})"
        (( delete )) && echo "then delete ${tar}"
        [[ -f "${marker}" ]] && echo "already extracted: nothing to do"
        return 0
    fi
    require_job fetch-gen4 plgrid "use --dry-run to inspect"
    umask 077
    if [[ -f "${marker}" ]]; then
        echo "complete  ${GEN4_ROOT}/${split} (${marker})"
        return 0
    fi
    fetch_one "${url}" "${tar}" "${GEN4_BYTES[${split}]}" || exit 1
    local crc
    echo "computing CRC32 of ${tar}"
    crc="$(python3 -I -c "${CRC_PY}" "${tar}")"
    if [[ "${crc}" != "${GEN4_CRC32[${split}]}" ]]; then
        echo "FAILED    ${tar}: CRC32 ${crc}, expected ${GEN4_CRC32[${split}]}; delete it and fetch again" >&2
        exit 1
    fi
    echo "verified  ${tar} CRC32 ${crc}"
    mkdir -p "${GEN4_ROOT}"
    tar -xf "${tar}" -C "${GEN4_ROOT}" --no-same-owner
    printf 'crc32 %s\nsource %s\n' "${crc}" "${url}" > "${marker}"
    echo "extracted ${GEN4_ROOT}/${split}: $(find "${GEN4_ROOT}/${split}" -type f | wc -l) files"
    if (( delete )); then
        rm -f "${tar}"
        echo "deleted   ${tar}"
    fi
}

# --------------------------------------------------------------------------- DSEC-MOT

cmd_layout_dsec_mot() {
    local dry=0
    [[ "${1:-}" == "--dry-run" ]] && dry=1 && shift
    [[ $# -eq 0 ]] || die "usage: layout-dsec-mot [--dry-run]"
    require_var LEMOT_DSEC_MOT_ANNOTATIONS
    set_paths
    local source split file seq target stamps found
    source="$(realpath -e "${LEMOT_DSEC_MOT_ANNOTATIONS}")" || die "${LEMOT_DSEC_MOT_ANNOTATIONS} does not exist"
    (( dry )) || umask 077
    found=0
    for split in train test; do
        for file in "${source}/${split}"/*.txt; do
            [[ -f "${file}" ]] || continue
            seq="$(basename "${file}" .txt)"
            target="${DSEC_MOT_ROOT}/annotations/${split}/${seq}.txt"
            stamps="$(compgen -G "${DSEC_ROOT}/*/${seq}/${seq}_image_timestamps.txt" || true)"
            [[ -n "${stamps}" && "$(wc -l <<< "${stamps}")" -eq 1 ]] ||
                die "need exactly one ${seq}_image_timestamps.txt under ${DSEC_ROOT}/*/${seq}/ (run fetch-dsec)"
            found=$((found + 1))
            if (( dry )); then
                echo "${file} -> ${target}"
                echo "${stamps} -> ${DSEC_MOT_ROOT}/${split}/${seq}/"
                continue
            fi
            if [[ -f "${target}" ]] && ! cmp -s "${file}" "${target}"; then
                die "${target} differs from ${file}; refusing to overwrite annotations"
            fi
            # The source may already be the annotation directory of the DSEC-MOT root.
            [[ "${file}" -ef "${target}" ]] || install -D -m 600 "${file}" "${target}"
            install -D -m 600 "${stamps}" "${DSEC_MOT_ROOT}/${split}/${seq}/${seq}_image_timestamps.txt"
        done
    done
    (( found > 0 )) || die "no annotation files in ${source}/{train,test}/"
    echo "DSEC-MOT root ${DSEC_MOT_ROOT}: ${found} sequences"
}

# --------------------------------------------------------------------------- environments

cmd_venv() {
    local kind="${1:-}" dry=0
    [[ "${2:-}" == "--dry-run" ]] && dry=1
    case "${kind}" in
        gh200) local arch=aarch64 modules=("${GH200_MODULES[@]}") extra=() ;;
        cpu) local arch=x86_64 modules=("${CPU_MODULES[@]}") extra=(-r "${REPO_ROOT}/requirements.txt" "${CPU_EXTRA_PINS[@]}") ;;
        *) die "usage: venv gh200|cpu [--dry-run]" ;;
    esac
    set_paths
    local venv="${VENV_ROOT}/large_event_mot_${kind}"
    local install=(python -m pip install --no-cache-dir --require-virtualenv -r "${REQUIREMENTS}" "${extra[@]}")
    if (( dry )); then
        echo "module add ${modules[*]}"
        echo "python -m venv ${venv}"
        echo "${install[*]}"
        return 0
    fi
    require_job "venv ${kind}" "$([[ ${kind} == gh200 ]] && echo plgrid-gpu-gh200 || echo plgrid)" "use --dry-run to inspect"
    require_arch "${arch}" "venv ${kind}"
    umask 077
    load_modules "${modules[@]}"
    [[ -d "${venv}" ]] || python -m venv "${venv}"
    activate "${venv}"
    "${install[@]}"
    python -m pip freeze > "${venv}/pip_freeze.txt"
    echo "environment ${venv} ready; package list in ${venv}/pip_freeze.txt"
}

cmd_preflight_cpu() {
    set_paths
    require_job preflight-cpu plgrid
    require_arch x86_64 preflight-cpu
    load_modules "${CPU_MODULES[@]}"
    activate "${VENV_ROOT}/large_event_mot_cpu"
    RVT_ROOT="${RVT_ROOT}" RVT_CHECKPOINT="${RVT_CHECKPOINT}" \
        STORAGE_PATHS="${ROOT}" \
        bash "${REPO_ROOT}/scripts/helios_event_mot_preflight.sh" --cpu-only \
        --output-dir "${RUNS_ROOT}/preflight_cpu/$(hostname -s)_$(date +%Y%m%d_%H%M%S)"
}

# --------------------------------------------------------------------------- runner

runner() {
    set_paths
    if [[ -n "${LEMOT_PYTHON:-}" ]]; then
        local python="${LEMOT_PYTHON}"
    else
        require_arch x86_64 "the runner (login node)"
        load_modules "${CPU_MODULES[@]}"
        activate "${VENV_ROOT}/large_event_mot_cpu"
        local python=python
    fi
    umask 077
    cd "${REPO_ROOT}"
    exec "${python}" -m src.experiments.large_event_mot "$@"
}

cmd_init() {
    require_var LEMOT_GPU_ACCOUNT LEMOT_CPU_ACCOUNT
    set_paths
    runner init \
        --runs-root "${RUNS_ROOT}" \
        --results-root "${RESULTS_ROOT}" \
        --manifest-root "${MANIFEST_ROOT}" \
        --cache-root "${CACHE_ROOT}" \
        --gen4-root "${GEN4_ROOT}" \
        --dsec-root "${DSEC_ROOT}" \
        --dsec-detection-labels-root "${DSEC_DETECTION_LABELS}" \
        --dsec-mot-root "${DSEC_MOT_ROOT}" \
        --rvt-root "${RVT_ROOT}" \
        --rvt-checkpoint "${RVT_CHECKPOINT}" \
        --venv-root "${VENV_ROOT}" \
        --trackeval-root "${TRACKEVAL_ROOT}" \
        --gpu-account "${LEMOT_GPU_ACCOUNT}" \
        --cpu-account "${LEMOT_CPU_ACCOUNT}" \
        --seed "${LEMOT_SEED:-0}" \
        "$@"
}

cmd_paths() {
    set_paths
    cat <<EOF
LEMOT_ROOT                  ${ROOT}
1Mpx release (gen4_root)    ${GEN4_ROOT}/{train,val}/
1Mpx tar downloads          ${GEN4_TAR_DIR}/
DSEC events (dsec_root)     ${DSEC_ROOT}/{train,test}/<seq>/events_left/
DSEC-Detection labels       ${DSEC_DETECTION_LABELS}/
DSEC-MOT root               ${DSEC_MOT_ROOT}/{annotations,train,test}/
RVT checkout                ${RVT_ROOT} (${RVT_COMMIT})
TrackEval checkout          ${TRACKEVAL_ROOT} (${TRACKEVAL_COMMIT})
RVT-B checkpoint            ${RVT_CHECKPOINT}
virtual environments        ${VENV_ROOT}/large_event_mot_{gh200,cpu}
sequence manifests          ${MANIFEST_ROOT}/
EF/VG caches                ${CACHE_ROOT}/
runs and experiment state   ${RUNS_ROOT}/ (state in ${RUNS_ROOT}/experiment/)
results                     ${RESULTS_ROOT}/
GPU account                 ${LEMOT_GPU_ACCOUNT:-<unset>}
CPU account                 ${LEMOT_CPU_ACCOUNT:-<unset>}
DSEC-MOT annotations source ${LEMOT_DSEC_MOT_ANNOTATIONS:-<unset>}
seed                        ${LEMOT_SEED:-0}
EOF
}

main() {
    local command="${1:-}"
    [[ $# -gt 0 ]] && shift
    case "${command}" in
        paths) cmd_paths "$@" ;;
        fetch-code) cmd_fetch_code "$@" ;;
        fetch-dsec) cmd_fetch_dsec "$@" ;;
        fetch-gen4) cmd_fetch_gen4 "$@" ;;
        layout-dsec-mot) cmd_layout_dsec_mot "$@" ;;
        venv) cmd_venv "$@" ;;
        preflight-cpu) cmd_preflight_cpu "$@" ;;
        init) cmd_init "$@" ;;
        plan | status | submit | reconcile | charge)
            set_paths
            runner "${command}" --runs-root "${RUNS_ROOT}" "$@"
            ;;
        help | -h | --help) usage ;;
        "") usage >&2; exit 2 ;;
        *) echo "unknown command: ${command}" >&2; usage >&2; exit 2 ;;
    esac
}

main "$@"
