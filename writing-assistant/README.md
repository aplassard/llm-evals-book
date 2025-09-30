# Writing Assistant Automation

This directory contains the automation script that converts walking audio notes into actionable writing artifacts and handles the end-to-end GitHub workflow.

## Overview

`transcribe_and_commit.sh` performs the following sequence:

1. Selects an audio file (latest in `~/Dropbox/walking_notes/audio` by default).
2. Runs `whisper` to transcribe the audio to text.
3. Generates a concise slug for naming via OpenRouter (`openai/gpt-5-mini`).
4. Saves the raw transcript into Dropbox (`~/Dropbox/walking_notes/transcripts/`) and this repository (`raw_notes/`).
5. Calls OpenRouter again to produce structured JSON notes with:
   - `text_summary`: cleaned summary of the recording.
   - `notes`: bullet-style sentences of the key ideas.
   - `articles_to_find`: references to gather, each with a `name`, `details`, and `status` (`known` or `unknown`).
   - `topics_to_review`: follow-up themes, each with a `topic` and supporting detail strings.
6. Commits both raw and cleaned files on a new worktree branch based on `main`, pushes it, and opens a pull request with GitHub CLI.

## Prerequisites

Install or ensure access to these tools:

- [`whisper`](https://github.com/openai/whisper) CLI (or another compatible Whisper executable in `$PATH`).
- [`jq`](https://stedolan.github.io/jq/) for JSON parsing.
- `git` with worktree support.
- [`gh`](https://cli.github.com/) authenticated for pushing branches and opening PRs.
- `curl` for OpenRouter API calls.
- Network access to https://openrouter.ai.

## Environment Setup

Place an `.env` file at the repository root (`../.env` relative to this directory) with at least:

```
OPENROUTER_API_KEY=sk-or-...
```

The script sources this file and fails fast if the key is missing. Never commit the `.env` file.

## Default Paths and Overrides

- Audio input: `~/Dropbox/walking_notes/audio` (`AUDIO_DIR` to override).
- Dropbox transcripts: `~/Dropbox/walking_notes/transcripts` (`TRANSCRIPTS_DIR`).
- Repository raw notes: `../raw_notes` (fixed relative to repo root).
- Repository cleaned notes: `../cleaned_notes` (fixed).
- Base branch for worktree: `main` (`BASE_BRANCH`).

Set any of these variables in the environment before running if you need custom locations.

## Usage

From the repository root:

```
./writing-assistant/transcribe_and_commit.sh [options] [AUDIO_FILE]
```

Options:

- `--log-file PATH` – append detailed log lines (with timestamps) to `PATH`.
- `--verbose` – echo log lines to stderr in addition to any file logging.
- `-h`, `--help` – display usage information.

If `AUDIO_FILE` is omitted, the newest file in the audio directory is used. The script aborts if the Git working tree is dirty or if the target branch already exists.

## Git Workflow Details

- A temporary worktree is created from the base branch and removed automatically after completion.
- Branch and filename follow the `<timestamp>-<slug>` convention.
- The commit message is `Add walking note transcript: <summary>`.
- A PR is opened with GitHub CLI; review the CLI output for the PR URL.

## Troubleshooting

- Ensure all required commands are available: run `command -v whisper jq git gh curl`.
- Whisper failures usually stem from missing models or unsupported file formats.
- OpenRouter errors often mean the API key is invalid/expired or network access is blocked.
- If the script stops before Git steps, check the log (if enabled) or terminal output for the exact failure message.
