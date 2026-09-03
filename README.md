# presentations

Landing page for <https://pc-piyush.github.io/presentations/> plus the individual
talk decks (reveal.js). The landing page uses the DFB theme shared with the main
site; each deck keeps its own self-contained project and styling.

## Build

```sh
./build.sh              # landing page + all decks -> _site/
./build.sh --publish    # then deploy to the gh-pages branch
```

`quarto preview` on its own only rebuilds the landing page (`index.qmd`); use
`build.sh` to (re)render the decks. Add a new talk by dropping its folder in,
adding a line to `talks.yml`, and adding its `.qmd` to the `DECKS` list in
`build.sh`.
