#!/usr/bin/env bash
#
# Build the presentations site: the DFB-themed landing page plus every reveal.js
# deck. Decks are listed in `render:` in _quarto.yml, so a single `quarto render`
# builds everything and drops each deck into _site/<dir>/.
#
# Usage:
#   ./build.sh            # build into _site/
#   ./build.sh --publish  # build, then `quarto publish gh-pages --no-render`

set -eu

echo ">> Rendering landing page + decks"
quarto render

if [ "${1:-}" = "--publish" ]; then
  echo ">> Publishing to gh-pages"
  quarto publish gh-pages --no-render --no-prompt
fi

echo ">> Done. Open _site/index.html"
