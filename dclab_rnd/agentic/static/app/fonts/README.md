# Fonts

The product serves its own fonts, so a page load makes no request to a font host and works offline.
Each is a variable font, so one file covers every weight the stylesheet uses. All are under the SIL Open
Font License 1.1 (the licence text sits next to each family).

| Family | Files | Source |
|---|---|---|
| Inter | `inter-latin.woff2`, `inter-latin-ext.woff2` | `@fontsource-variable/inter` 5.3.0 (Inter 4, rsms/inter) |
| Vazirmatn | `vazirmatn-arabic.woff2`, `vazirmatn-latin.woff2` | `@fontsource-variable/vazirmatn` 5.3.0 (rastikerdar/vazirmatn) |
| JetBrains Mono | `jetbrains-mono-latin.woff2` | JetBrains Mono 2.304 (JetBrains/JetBrainsMono), subset to Latin, punctuation, arrows and math signs with `fontTools.subset` (layout features kept) |

Scripts that are not Latin or Arabic (Cyrillic, Greek, Vietnamese) fall back to the system font.
`python -m dclab_rnd.agentic.web.build` copies this folder to `static/app/fonts/`; the `@font-face` rules are at the top of `styles.css`.
