#!/usr/bin/env bash
set -euo pipefail

# Abort if required tools are missing early.
for cmd in whisper jq git gh curl; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found in PATH." >&2
    exit 1
  fi
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
audio_dir="${AUDIO_DIR:-$HOME/Dropbox/walking_notes/audio}"
transcripts_dir="${TRANSCRIPTS_DIR:-$HOME/Dropbox/walking_notes/transcripts}"
raw_notes_dir="${RAW_NOTES_DIR:-$repo_root/raw_notes}"
env_file="${repo_root}/.env"

if [[ ! -f "$env_file" ]]; then
  echo "Error: .env file not found at $env_file" >&2
  exit 1
fi

# Load secrets without polluting environment unintentionally.
set -a
# shellcheck disable=SC1090
source "$env_file"
set +a

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Error: OPENROUTER_API_KEY not set in environment." >&2
  exit 1
fi

mkdir -p "$transcripts_dir"
mkdir -p "$raw_notes_dir"

audio_file="${1:-}"
if [[ -z "$audio_file" ]]; then
  if [[ ! -d "$audio_dir" ]]; then
    echo "Error: audio directory $audio_dir not found." >&2
    exit 1
  fi
  audio_file=$(ls -t "$audio_dir"/* 2>/dev/null | head -n1 || true)
  if [[ -z "$audio_file" ]]; then
    echo "Error: no audio files found in $audio_dir." >&2
    exit 1
  fi
fi

if [[ ! -f "$audio_file" ]]; then
  echo "Error: audio file $audio_file does not exist." >&2
  exit 1
fi

# Ensure repo is clean before making changes.
cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: git working tree is dirty. Commit or stash changes before running." >&2
  exit 1
fi

timestamp=$(date +%s)

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT

whisper_output_prefix="$work_dir/transcript"

if ! whisper "$audio_file" \
  --output_dir "$work_dir" \
  --output_format txt \
  --verbose False >/dev/null 2>&1; then
  echo "Error: whisper transcription failed." >&2
  exit 1
fi

transcript_file=$(find "$work_dir" -maxdepth 1 -type f -name '*.txt' | head -n1)
if [[ -z "$transcript_file" ]]; then
  echo "Error: transcript text file not produced by whisper." >&2
  exit 1
fi

transcript_dest="$transcripts_dir/${timestamp}.txt"
cp "$transcript_file" "$transcript_dest"

transcript_excerpt=$(head -c 6000 "$transcript_file" | tr '\r' '\n')

payload=$(jq -n \
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

response=$(curl -sS -f https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${OPENROUTER_API_KEY}" \
  -d "$payload")

summary=$(echo "$response" | jq -r '.choices[0].message.content' | tr -d '\r')
summary_line=$(echo "$summary" | head -n1 | sed 's/^\s\+//; s/\s\+$//')

if [[ -z "$summary_line" || "$summary_line" == "null" ]]; then
  echo "Error: failed to obtain summary from OpenRouter response." >&2
  exit 1
fi

slug=$(echo "$summary_line" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9]/-/g' -e 's/-\+/-/g' -e 's/^-//' -e 's/-$//')
if [[ -z "$slug" ]]; then
  slug="note"
fi

filename="${timestamp}-${slug}.txt"
branch_name="${timestamp}-${slug}"

mv "$transcript_dest" "$transcripts_dir/$filename"
transcript_dest="$transcripts_dir/$filename"
raw_notes_path="$raw_notes_dir/$filename"
cp "$transcript_dest" "$raw_notes_path"

git checkout -b "$branch_name"

git add "raw_notes/$filename"
commit_msg="Add walking note transcript: $summary_line"
if ! git commit -m "$commit_msg"; then
  echo "Error: git commit failed." >&2
  exit 1
fi

git push -u origin "$branch_name"

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
fi

echo "Transcript saved to $transcript_dest"
echo "Transcript copied to $raw_notes_path"
echo "Branch $branch_name created and pushed with PR title: $pr_title"
echo "Summary: $summary_line"
