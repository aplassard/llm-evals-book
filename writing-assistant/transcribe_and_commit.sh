#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--log-file PATH] [--verbose] [AUDIO_FILE]

Options:
  --log-file PATH   Append detailed progress logs to PATH.
  --verbose         Mirror log output to stderr.
  -h, --help        Show this help message.

If AUDIO_FILE is not provided, the newest file in the default audio directory is used.
USAGE
}

# Logging helpers
LOG_FILE=""
VERBOSE=0
log_info() {
  local msg="$1"
  local timestamp
  timestamp=$(date '+%Y-%m-%dT%H:%M:%S%z')
  if [[ -n "$LOG_FILE" ]]; then
    echo "[$timestamp] $msg" >> "$LOG_FILE"
  fi
  if [[ $VERBOSE -eq 1 ]]; then
    echo "[$timestamp] $msg" >&2
  fi
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

audio_arg=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-file)
      if [[ $# -lt 2 ]]; then
        echo "Error: --log-file requires a path argument." >&2
        exit 1
      fi
      LOG_FILE=$2
      shift 2
      ;;
    --log-file=*)
      LOG_FILE="${1#*=}"
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "Error: unknown option $1" >&2
      exit 1
      ;;
    *)
      if [[ -n "$audio_arg" ]]; then
        echo "Error: multiple positional arguments provided." >&2
        exit 1
      fi
      audio_arg="$1"
      shift
      ;;
  esac
done

if [[ -n "$LOG_FILE" ]]; then
  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"
fi

log_info "Starting transcription workflow."

# Abort if required tools are missing early.
for cmd in whisper jq git gh curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found in PATH." >&2
    log_info "Missing required command: $cmd"
    exit 1
  fi
done

log_info "All required commands located."

audio_dir="${AUDIO_DIR:-$HOME/Dropbox/walking_notes/audio}"
transcripts_dir="${TRANSCRIPTS_DIR:-$HOME/Dropbox/walking_notes/transcripts}"
raw_notes_dir="$repo_root/raw_notes"
cleaned_notes_dir="$repo_root/cleaned_notes"
env_file="${repo_root}/.env"
base_branch="${BASE_BRANCH:-main}"

log_info "Using base branch: $base_branch"

if [[ ! -f "$env_file" ]]; then
  echo "Error: .env file not found at $env_file" >&2
  log_info "Missing .env file."
  exit 1
fi

# Load secrets without polluting environment unintentionally.
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY not set in environment." >&2
  log_info "OPENROUTER_API_KEY not present after sourcing .env."
  exit 1
fi

mkdir -p "$transcripts_dir"
mkdir -p "$raw_notes_dir"
mkdir -p "$cleaned_notes_dir"

log_info "Ensured transcript, raw notes, and cleaned notes directories exist."

audio_file="$audio_arg"
if [[ -z "$audio_file" ]]; then
  if [[ ! -d "$audio_dir" ]]; then
    echo "Error: audio directory $audio_dir not found." >&2
    log_info "Audio directory missing: $audio_dir"
    exit 1
  fi
  audio_file=$(ls -t "$audio_dir"/* 2>/dev/null | head -n1 || true)
  if [[ -z "$audio_file" ]]; then
    echo "Error: no audio files found in $audio_dir." >&2
    log_info "No audio files found in $audio_dir"
    exit 1
  fi
fi

if [[ ! -f "$audio_file" ]]; then
  echo "Error: audio file $audio_file does not exist." >&2
  log_info "Provided audio file does not exist: $audio_file"
  exit 1
fi

log_info "Selected audio file: $audio_file"

# Ensure base branch exists.
if ! git -C "$repo_root" rev-parse --verify "$base_branch" >/dev/null 2>&1; then
  echo "Error: base branch '$base_branch' not found." >&2
  log_info "Base branch $base_branch missing."
  exit 1
fi

# Ensure repo is clean before making changes.
if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then
  echo "Error: git working tree is dirty. Commit or stash changes before running." >&2
  log_info "Repository dirty; aborting."
  exit 1
fi

log_info "Repository clean. Proceeding with transcription."

timestamp=$(date +%s)
log_info "Using timestamp: $timestamp"

work_dir=$(mktemp -d)
git_worktree_dir=""

cleanup() {
  local exit_code="$1"
  if [[ -d "$work_dir" ]]; then
    rm -rf "$work_dir"
  fi
  if [[ -n "$git_worktree_dir" ]]; then
    if git -C "$repo_root" worktree list | grep -Fq " $git_worktree_dir"; then
      git -C "$repo_root" worktree remove --force "$git_worktree_dir" >/dev/null 2>&1 || true
    fi
    rm -rf "$git_worktree_dir" 2>/dev/null || true
  fi
  log_info "Cleanup complete with exit code $exit_code."
  exit "$exit_code"
}
trap 'cleanup "$?"' EXIT

log_info "Temporary transcription directory: $work_dir"

if ! whisper "$audio_file" \
  --output_dir "$work_dir" \
  --output_format txt \
  --verbose False >/dev/null 2>&1; then
  echo "Error: whisper transcription failed." >&2
  log_info "Whisper transcription failed."
  exit 1
fi

log_info "Whisper transcription completed."

transcript_file=$(find "$work_dir" -maxdepth 1 -type f -name '*.txt' | head -n1)
if [[ -z "$transcript_file" ]]; then
  echo "Error: transcript text file not produced by whisper." >&2
  log_info "No transcript file produced by whisper."
  exit 1
fi

log_info "Transcript file located: $transcript_file"

transcript_dest="$transcripts_dir/${timestamp}.txt"
cp "$transcript_file" "$transcript_dest"

log_info "Copied transcript to Dropbox path: $transcript_dest"

transcript_excerpt=$(head -c 6000 "$transcript_file" | tr '\r' '\n')

payload_summary=$(jq -n \
  --arg system "You create concise, human-readable slugs for walking note transcripts. Respond with 3-6 plain words suitable for a filename, avoid punctuation." \
  --arg user "Transcript excerpt:\n$transcript_excerpt" \
  '{
    model: "openai/gpt-5-mini",
    temperature: 0.2,
    messages: [
      {role: "system", content: $system},
      {role: "user", content: $user}
    ]
  }')

log_info "Requesting slug from OpenRouter."

summary_response=$(curl -sS -f https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
  -d "$payload_summary")

summary=$(echo "$summary_response" | jq -r '.choices[0].message.content' | tr -d '\r')
summary_line=$(echo "$summary" | head -n1 | sed 's/^\s\+//; s/\s\+$//')

if [[ -z "$summary_line" || "$summary_line" == "null" ]]; then
  echo "Error: failed to obtain summary from OpenRouter response." >&2
  log_info "Summary response invalid."
  exit 1
fi

log_info "Received summary line: $summary_line"

slug=$(echo "$summary_line" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9]/-/g' -e 's/-\+/-/g' -e 's/^-//' -e 's/-$//')
if [[ -z "$slug" ]]; then
  slug="note"
fi

log_info "Generated slug: $slug"

filename="${timestamp}-${slug}.txt"
branch_name="${timestamp}-${slug}"

log_info "Planned filename: $filename"
log_info "Branch name: $branch_name"

mv "$transcript_dest" "$transcripts_dir/$filename"
transcript_dest="$transcripts_dir/$filename"
cp "$transcript_dest" "$raw_notes_dir/$filename"

log_info "Transcript renamed and copied to raw notes: $raw_notes_dir/$filename"

transcript_contents=$(cat "$transcript_file")

payload_notes=$(jq -n \
  --arg system "You are processing a transcription of impromptu spoken walking notes. Produce structured planning artifacts to turn the raw transcript into actionable writing material. Respond ONLY with valid JSON using this shape: {\"text_summary\": string, \"notes\": [string], \"articles_to_find\": [{\"name\": string, \"details\": string, \"status\": \"known\"|\"unknown\"}], \"topics_to_review\": [{\"topic\": string, \"details\": [string]}]}. Make the summary concise but faithful to the audio. Notes must be cleaned sentences capturing core ideas. Articles should include whether the reference is known (explicitly named) or unknown (needs research) and what to look for. Topics should gather related follow-up ideas in their detail lists. Use Markdown-free plain text in strings." \
  --arg transcript "This is a transcription of audio walking notes. Convert it into useful references for writing projects. Transcript:\n$transcript_contents" \
  '{
    model: "openai/gpt-5-mini",
    temperature: 0.2,
    response_format: {type: "json_object"},
    messages: [
      {role: "system", content: $system},
      {role: "user", content: $transcript}
    ]
  }')

log_info "Requesting structured notes from OpenRouter."

notes_response=$(curl -sS -f https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
  -d "$payload_notes")

notes_content=$(echo "$notes_response" | jq -r '.choices[0].message.content' | tr -d '\r')
if [[ -z "$notes_content" || "$notes_content" == "null" ]]; then
  echo "Error: failed to obtain structured notes from OpenRouter response." >&2
  log_info "Structured notes response invalid."
  exit 1
fi

cleaned_filename="${filename%.txt}.json"
cleaned_path="$cleaned_notes_dir/$cleaned_filename"

if ! echo "$notes_content" | jq '.' > "$cleaned_path"; then
  echo "Error: structured notes are not valid JSON." >&2
  log_info "Structured notes JSON invalid."
  exit 1
fi

required_keys=(text_summary notes articles_to_find topics_to_review)
for key in "${required_keys[@]}"; do
  if ! jq -e --arg key "$key" 'has($key)' < "$cleaned_path" >/dev/null; then
    echo "Error: structured notes missing required key '$key'." >&2
    log_info "Structured notes missing key: $key"
    exit 1
  fi
done

log_info "Structured notes saved to $cleaned_path"

# Prepare git worktree based on main
log_info "Setting up git worktree for branch $branch_name."

if git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch_name"; then
  echo "Error: branch '$branch_name' already exists." >&2
  log_info "Branch already exists: $branch_name"
  exit 1
fi

git_worktree_dir=$(mktemp -d "${TMPDIR:-/tmp}/transcribe-worktree.XXXXXX")
rmdir "$git_worktree_dir"

if ! git -C "$repo_root" worktree add -b "$branch_name" "$git_worktree_dir" "$base_branch" >/dev/null; then
  echo "Error: failed to create worktree for branch '$branch_name'." >&2
  log_info "git worktree add failed for $branch_name"
  exit 1
fi

log_info "Git worktree created at $git_worktree_dir"

worktree_raw_notes_dir="$git_worktree_dir/raw_notes"
worktree_cleaned_notes_dir="$git_worktree_dir/cleaned_notes"

mkdir -p "$worktree_raw_notes_dir" "$worktree_cleaned_notes_dir"
cp "$transcript_dest" "$worktree_raw_notes_dir/$filename"
cp "$cleaned_path" "$worktree_cleaned_notes_dir/$cleaned_filename"

log_info "Copied files into worktree."

pushd "$git_worktree_dir" >/dev/null

git add "raw_notes/$filename" "cleaned_notes/$cleaned_filename"
commit_msg="Add walking note transcript: $summary_line"
if ! git commit -m "$commit_msg" >/dev/null; then
  echo "Error: git commit failed." >&2
  log_info "git commit failed."
  popd >/dev/null
  exit 1
fi

log_info "Commit created: $commit_msg"

if ! git push -u origin "$branch_name" >/dev/null; then
  echo "Error: git push failed." >&2
  log_info "git push failed."
  popd >/dev/null
  exit 1
fi

log_info "Branch pushed to origin/$branch_name"

pr_title="Add walking note transcript ${timestamp}"
pr_body=$(cat <<BODY
## Summary
- Source audio: $audio_file
- Summary slug: $summary_line
- Timestamp: $timestamp
BODY
)

if ! gh pr create --title "$pr_title" --body "$pr_body" --head "$branch_name"; then
  echo "Warning: Pull request creation failed. Review manually." >&2
  log_info "gh pr create failed."
else
  log_info "Pull request created with title: $pr_title"
fi

popd >/dev/null

log_info "Git operations complete."

echo "Transcript saved to $transcript_dest"
echo "Transcript copied to $raw_notes_dir/$filename"
echo "Structured notes saved to $cleaned_path"
echo "Branch $branch_name created and pushed with PR title: $pr_title"
echo "Summary: $summary_line"

if [[ -n "$LOG_FILE" ]]; then
  echo "Logs written to $LOG_FILE"
fi
