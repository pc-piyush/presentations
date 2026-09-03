#!/usr/bin/env bash
#
# Build the presentations site: the DFB-themed landing page plus every talk deck.
# Each deck under this repo is its own self-contained Quarto project, so the root
# `quarto render` (restricted to index.qmd in _quarto.yml) does not touch them —
# we render each one here and drop its output into _site/<dir>/.
#
# Usage:
#   ./build.sh            # build into _site/
#   ./build.sh --publish  # build, then `quarto publish gh-pages --no-render`

set -u

DECKS=(
  "GMS6804_20260217/slides.qmd"
  "GMS6805_20260226/slides.qmd"
  "GMS6805_FCT_CDM/presentation_20260416.qmd"
  "HOBISharkTank_2026/pitch_2026.qmd"
  "F26_GMS7858/ibi-hypertension.qmd"
  "JournalClub/2026-03-23-RAG/RAG.qmd"
)

echo ">> Rendering landing page"
quarto render || { echo "landing page render failed"; exit 1; }

for f in "${DECKS[@]}"; do
  dir=$(dirname "$f")
  stem=$(basename "${f%.qmd}")
  echo ">> Rendering deck: $f"
  if ! quarto render "$f"; then
    echo "!! skipped $f (render failed — check its dependencies)"
    continue
  fi
  mkdir -p "_site/$dir"
  cp "$dir/$stem.html" "_site/$dir/"
  [ -d "$dir/${stem}_files" ] && cp -R "$dir/${stem}_files" "_site/$dir/"
  [ -d "$dir/images" ]        && cp -R "$dir/images"        "_site/$dir/"
done

if [ "${1:-}" = "--publish" ]; then
  echo ">> Publishing to gh-pages"
  quarto publish gh-pages --no-render --no-prompt
fi

echo ">> Done. Open _site/index.html"
