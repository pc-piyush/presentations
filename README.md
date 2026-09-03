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

1. Drop the deck's folder in as `<folder>/index.qmd` (plus its `images/` etc.).
2. Give `index.qmd` front matter `title`, `date`, `description`, `categories`
   (these feed the landing-page card) and `brand: false`.

That's it — `render: ["index.qmd", "*/index.qmd"]` and the `*/index.qmd` listing
glob in `index.qmd` pick it up automatically.
