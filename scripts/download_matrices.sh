#!/bin/bash
# Download every SuiteSparse matrix benchmarked by this project into ./matrices.
#
# The R-MAT inputs are deliberately absent: they are not SuiteSparse matrices
# and are produced locally by scripts/generate_rmat.sh.
#
# This is not a batch script. Run it on a node with outbound internet access,
# which on most clusters means the login node and not a compute node.
#
#   scripts/download_matrices.sh                  # everything
#   scripts/download_matrices.sh eu-2005 rajat31  # only the named matrices
#
# Completed matrices are skipped, and interrupted transfers resume, so the
# script can be re-run safely. Set KEEP_ARCHIVES=1 to retain the .tar.gz files.

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

BASE_URL="${BASE_URL:-https://suitesparse-collection-website.herokuapp.com/MM}"
DEST="${DEST:-matrices}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

# Group/name exactly as published by the SuiteSparse Matrix Collection. The
# HTTPS entry point redirects to a plain-HTTP file host, which is what the
# collection's own download links do; wget follows it.
#
# Ordered smallest-first so a partial run still yields usable inputs. Compressed
# sizes are 11 MB, 123 MB, 46 MB, 79 MB, 70 MB, 1.3 GB, and 2.5 GB, and Matrix
# Market text expands roughly sixfold, so nlpkkt240 and webbase-2001 need tens
# of GB of free space between them.
MATRICES=(
    Williams/webbase-1M
    Freescale/FullChip
    LAW/eu-2005
    Rajat/rajat31
    PARSEC/Si41Ge41H72
    Schenk/nlpkkt240
    LAW/webbase-2001
)

if ! command -v wget >/dev/null 2>&1; then
    echo "wget is required but was not found in PATH" >&2
    exit 1
fi

if (( $# > 0 )); then
    SELECTED=()
    for WANTED in "$@"; do
        MATCH=""
        for ENTRY in "${MATRICES[@]}"; do
            if [[ "$ENTRY" == "$WANTED" || "${ENTRY##*/}" == "$WANTED" ]]; then
                MATCH="$ENTRY"
                break
            fi
        done
        if [[ -z "$MATCH" ]]; then
            echo "Unknown matrix: $WANTED" >&2
            echo "Available: ${MATRICES[*]##*/}" >&2
            exit 1
        fi
        SELECTED+=("$MATCH")
    done
    MATRICES=("${SELECTED[@]}")
fi

mkdir -p "$DEST"

for ENTRY in "${MATRICES[@]}"; do
    NAME="${ENTRY##*/}"
    TARGET="$DEST/$NAME.mtx"

    if [[ -s "$TARGET" ]]; then
        echo "== $NAME: already in $DEST, skipping"
        continue
    fi

    echo "== $NAME: downloading $BASE_URL/$ENTRY.tar.gz"
    # --continue resumes a transfer interrupted by a walltime or network drop.
    wget --continue --tries=3 --timeout=60 --progress=dot:giga \
        -P "$DEST" "$BASE_URL/$ENTRY.tar.gz"

    ARCHIVE="$DEST/$NAME.tar.gz"
    echo "== $NAME: extracting"
    # Every SuiteSparse Matrix Market archive stores <name>/<name>.mtx.
    tar -xzf "$ARCHIVE" -C "$DEST" "$NAME/$NAME.mtx"
    mv "$DEST/$NAME/$NAME.mtx" "$TARGET"
    rm -rf "${DEST:?}/$NAME"

    if [[ "$KEEP_ARCHIVES" == "1" ]]; then
        echo "== $NAME: keeping $ARCHIVE"
    else
        rm -f "$ARCHIVE"
    fi
done

echo
echo "Matrix Market inputs in $DEST:"
shopt -s nullglob
PRESENT=("$DEST"/*.mtx)
if (( ${#PRESENT[@]} == 0 )); then
    echo "  (none)"
else
    ls -lh "${PRESENT[@]}"
fi
