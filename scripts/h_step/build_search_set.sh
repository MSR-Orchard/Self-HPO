#!/bin/bash
# build_search_set.sh <args>: run build_search_set.py with the evaluation environment.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"
exec "$SELFHPO_EVAL_ENV/bin/python" "$HERE/build_search_set.py" "$@"
