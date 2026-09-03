# presentations

Landing page for <https://pc-piyush.github.io/presentations/> plus the individual
talk decks (reveal.js). The landing page uses the DFB theme shared with the main
site; each deck keeps its own front matter and theme (`brand: false` so the DFB
brand doesn't bleed in).

## Build

```sh
quarto preview          # landing page + decks, live reload
quarto render           # landing page + decks -> _site/
./build.sh --publish    # render, then deploy to the gh-pages branch
```

## Add a talk

1. Drop the deck's folder in (its `.qmd` plus `images/` etc.).
2. Give the `.qmd` front matter `title`, `date`, `description`, `categories`
   (these feed the landing-page card) and `brand: false`.
3. Add its `.qmd` path to `render:` in `_quarto.yml` **and** to `contents:` in
   `index.qmd`.
