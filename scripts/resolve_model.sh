# resolve_model <local dir | Hugging Face model id>: print a local model dir.
# A model id is downloaded to $SELFHPO_WORK/models/<name> (files already complete are skipped).
# Set PY to the python that has huggingface_hub (default python3).
resolve_model() {
  if [ -d "$1" ]; then realpath "$1"; return; fi
  local dst="${SELFHPO_WORK:?set SELFHPO_WORK}/models/$(basename "$1")"
  echo "downloading $1 to $dst ..." >&2
  "${PY:-python3}" -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], local_dir=sys.argv[2])" "$1" "$dst" >&2 \
    || { echo "ERR: $1 is neither a local dir nor a downloadable model id" >&2; return 1; }
  realpath "$dst"
}
