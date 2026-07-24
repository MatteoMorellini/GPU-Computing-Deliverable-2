#!/usr/bin/env bash

set -euo pipefail

generator=${1:-./bin/generate_rmat}
test_dir=$(mktemp -d /tmp/graph500-rmat-test.XXXXXX)
trap 'rm -rf -- "$test_dir"' EXIT

"$generator" \
    --scale 4 \
    --edge-factor 4 \
    --seed1 2 \
    --seed2 3 \
    --chunk-edges 7 \
    --output "$test_dir/symmetric-a.mtx"

"$generator" \
    --scale 4 \
    --edge-factor 4 \
    --seed1 2 \
    --seed2 3 \
    --chunk-edges 11 \
    --output "$test_dir/symmetric-b.mtx"

cmp "$test_dir/symmetric-a.mtx" "$test_dir/symmetric-b.mtx"

awk '
    /^%/ { next }
    !dimensions { rows = $1; cols = $2; declared = $3; dimensions = 1; next }
    {
        if ($1 < $2) {
            print "symmetric entry is outside the lower triangle" > "/dev/stderr"
            exit 1
        }
        if ($1 < 1 || $1 > rows || $2 < 1 || $2 > cols) {
            print "entry is outside the declared dimensions" > "/dev/stderr"
            exit 1
        }
        entries++
    }
    END {
        if (rows != 16 || cols != 16 || declared != 64 || entries != declared) {
            print "unexpected symmetric Matrix Market dimensions/count" > "/dev/stderr"
            exit 1
        }
    }
' "$test_dir/symmetric-a.mtx"

"$generator" \
    --scale 4 \
    --edges 64 \
    --format general \
    --drop-self-loops \
    --chunk-edges 9 \
    --output "$test_dir/general.mtx"

awk '
    /^%/ { next }
    !dimensions { rows = $1; cols = $2; declared = $3; dimensions = 1; next }
    {
        if ($1 == $2) {
            print "self-loop remained after --drop-self-loops" > "/dev/stderr"
            exit 1
        }
        entries++
    }
    END {
        if (rows != 16 || cols != 16 || entries != declared || declared > 64) {
            print "unexpected general Matrix Market dimensions/count" > "/dev/stderr"
            exit 1
        }
    }
' "$test_dir/general.mtx"

if "$generator" --scale 4 --edge-factor 4 --edges 64 \
    --output "$test_dir/conflicting.mtx" >/dev/null 2>&1; then
    echo "generator accepted conflicting size options" >&2
    exit 1
fi

if "$generator" --scale 4 --edges 1 \
    --output "$test_dir/general.mtx" >/dev/null 2>&1; then
    echo "generator overwrote an existing file without --force" >&2
    exit 1
fi

echo "Graph500 R-MAT generator tests passed"
