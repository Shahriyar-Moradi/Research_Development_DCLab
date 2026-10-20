# The TripsSource design kit, as copied here

Copied unchanged from Flowscope's `design/tripssource-kit/` at commit `89b13a3` (2026-10-19). Update it by copying the
folder again; never edit these files. The product's own rules sit after them, in `../styles.css` (the variables pointed
at the kit's tokens) and `../look.css` (the product's components drawn the kit's way, with `--ts-*` tokens only).

`python -m dclab_rnd.agentic.web.build` writes `ts-tokens.css`, `ts-theme-blue.css` and `ts-components.css`, in that order,
at the head of `static/app/app.css`. Their comments are left out of
app.css (one shows a font host's link; the app names no other host). The PDF reference, `TripsSource-Design-Kit.pdf`, is
the owner's (`Tripssource_UI/`, not committed); `IMPLEMENTING.md` and `README.md` here are the kit's own text.
