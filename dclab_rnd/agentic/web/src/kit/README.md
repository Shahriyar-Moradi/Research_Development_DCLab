# TripsSource design kit

The look of the TripsSource admin, written down so any site can use it: DCLab R&D, the DCLab product, Flowscope.
Everything here was read from the TripsSource admin's own stylesheets. The tokens and the component rules are exact.
The only rebuilt part is the app frame (sidebar column, top bar, canvas), because its stylesheet was not among the
ones the admin loads on its pages. It is marked as rebuilt wherever it appears.

| File | What it is |
| --- | --- |
| `ts-tokens.css` | Every design token (184 light, 71 dark overrides) as `--ts-*` custom properties, plus the fonts. |
| `ts-theme-blue.css` | Optional. Blue as the main colour and light green as the accent, in light and dark. Load after the tokens. |
| `ts-components.css` | 55 components and page layouts, 880 rules, renamed to plain `.ts-*` classes, plus the rebuilt frame. |
| `example.html` | Every component on one page, with switches for green or blue and light or dark. Open it in a browser. |
| `TripsSource-Design-Kit.pdf` | This spec as a 31-page PDF, every component drawn with the kit's own CSS. Rebuild it with `python3 design/tripssource-kit/pdf/build.py`. |
| `IMPLEMENTING.md` · `TripsSource-Implementation-Guide.pdf` | How to implement the kit in another product, step by step: install, frame, pages, components, checks, migrating an existing app, notes for the DCLab product and DCLab R&D, and a prompt for a coding agent. Rebuild the PDF with `python3 design/tripssource-kit/pdf/build_guide.py`. |

## Contents

1. [Setup](#1-setup)
2. [Principles](#2-principles)
3. [Tokens](#3-tokens)
4. [Typography](#4-typography)
5. [Layout: the frame](#5-layout-the-frame)
6. [Components](#6-components)
7. [Page patterns](#7-page-patterns)
8. [States, focus and motion](#8-states-focus-and-motion)
9. [The blue and light-green theme](#9-the-blue-and-light-green-theme)
10. [Using it in React or Next.js](#10-using-it-in-react-or-nextjs)
11. [Class reference](#11-class-reference)

---

## 1. Setup

```html
<html lang="en" data-ts-reset>           <!-- data-ts-reset: TripsSource's base and type utilities -->
<head>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap">
  <link rel="stylesheet" href="ts-tokens.css">
  <link rel="stylesheet" href="ts-theme-blue.css">   <!-- optional: blue + light green -->
  <link rel="stylesheet" href="ts-components.css">
</head>
```

- **Theme.** With no attribute, the system setting picks light or dark. `data-ts-theme="light"` or `data-ts-theme="dark"`
  on `<html>` forces one. TripsSource offers "Light / Dark / System" in a three-way switch (`.ts-colour-mode__*`).
- **Base.** `data-ts-reset` on `<html>` turns on:
  - `box-sizing: border-box` everywhere;
  - body text in Plus Jakarta Sans 13.5px in the text colour, antialiased (set the page background yourself: the
    frame's `.ts-app` uses the ground);
  - h1 to h6 in Outfit, semibold;
  - the selection colour (accent-soft), the focus ring and thin scrollbars;
  - the type utilities `.ts-text-*`, `.ts-weight-*`, `.ts-tracking-*`, `.ts-label-caps`, `.ts-numeric`,
    `.ts-font-display` and `.ts-font-body`.
- **Naming.** `.ts-<component>` is a component's root and `.ts-<component>__<part>` is one of its parts. Variants, sizes
  and states are attributes, exactly as in TripsSource:
  - `data-variant`, `data-size` (`sm` | `md` | `lg`), `data-tone`, `data-state`;
  - `data-invalid`, `data-caps`, `data-solid`, `data-selected`;
  - `aria-selected`, `aria-expanded`, `aria-current`, `aria-sort`, `aria-pressed`.
- **Only tokens.** No component uses a raw colour or size, so changing a token re-themes everything.

## 2. Principles

What makes this look what it is, in the order it matters:

1. **Calm ground, white cards.**
   - The page sits on a soft grey-green ground (`--ts-ground`).
   - Content sits in white cards with a 1px hairline border (`--ts-border`), a 22px radius and an almost invisible
     shadow (`0 1px 2px` at 4%).
   - There are no heavy shadows, except on overlays (modal, drawer, toast).
2. **Two typefaces with clear jobs.**
   - **Outfit** is for what you read first: page titles (38px), card titles (19px), big numbers (32px), modal titles.
   - **Plus Jakarta Sans** is for everything else, at 12.5 to 15px.
   - Small labels are bold, uppercase and widely tracked (`.12em`), never large.
3. **Round everything.**
   - Buttons, badges, chips, tabs, pagination and the search field are pills (`99px`).
   - Inputs and menus use 14px, cards 22px, the app canvas 30px.
   - Icon chips are rounded squares (14px).
4. **One accent, used sparingly.** The bright accent (lime in TripsSource, light green in the blue theme) appears only
   as a marker:
   - the 4px bar beside the current nav row;
   - the selected tab's underline;
   - the focus ring;
   - progress fills;
   - the hover border of link cards.

   Actions use the deep brand colour. The current row of a table or the nav uses the soft accent tint.
5. **Ink for "on".**
   - An open nav group, a selected pill tab, a toast, a tooltip and the save bar are deep ink (near-black navy-green)
     with white text.
   - This is the strongest emphasis in the system.
6. **Status is tinted, never loud.**
   - Each tone (success, warning, danger, info, plus neutral and violet) has five colours: solid, fg, bg, border and chip.
   - Badges use chip + fg. Alerts use bg + border + fg. Solid is only for a solid badge or a danger button.
7. **Generous, regular spacing.**
   - The scale is 4 / 8 / 10 / 12 / 14 / 18 / 22 / 26 / 32 px.
   - Cards pad 20px. Sections are 18 to 22px apart. The page inset is 16px.
8. **Controls come in three heights.** 34 (sm), 42 (md) and 46 (lg). Tabs are 38, nav rows 44, child nav rows 36.
   Everything on one line shares a height.

## 3. Tokens

All values are in `ts-tokens.css`. Light values are listed first, then dark where it differs.

### 3.1 Colour: surfaces, text, lines

| Token | Light | Dark | Used for |
| --- | --- | --- | --- |
| `--ts-ground` | `#e7e8e3` | `#12111f` | The page behind everything; the sidebar sits on it |
| `--ts-surface` | `#fff` | `#1b1a2b` | Cards, panels, tables, the active nav row, menus |
| `--ts-surface-sunken` | `#f6f7f3` | `#161526` | Table header row, facts, the app canvas, pill-tab track |
| `--ts-surface-muted` | `#f1f3ed` | `#222136` | Segmented-control track, icon chips, avatars, neutral wells |
| `--ts-surface-hover` | `#edefe8` | `#28273d` | Hover fill of rows, buttons, nav rows |
| `--ts-surface-input` | `#f8f9f5` | `#1e1d30` | Inputs, selects, textareas, view tiles |
| `--ts-surface-row-hover` | `#f8f9f5` | `#201f32` | Table row hover |
| `--ts-text` | `#1a2418` | `#ececf4` | Body text and titles |
| `--ts-text-secondary` | `#48524a` | `#bab9ce` | Labels, subtitles, nav rows, table headers |
| `--ts-text-muted` | `#7c8a7f` | `#9796ae` | Placeholders, meta |
| `--ts-text-faint` | `#9aa79d` | `#77768f` | Nav icons at rest, breadcrumb slashes, sort icons |
| `--ts-text-inverse` | `#fff` | `#fff` | Text on ink and on solid fills |
| `--ts-border` | `#edeee8` | `#2a2940` | Card and cell hairlines |
| `--ts-border-strong` | `#e3e7dc` | `#34334c` | Inputs, secondary buttons, chips, pagination |
| `--ts-scrim-modal` / `--ts-scrim-drawer` | `#1a24185c` / `#1a241852` | `#0000009e` / `#0000008c` | Overlays |

### 3.2 Colour: brand, accent, ink

| Token | Light | Dark | Used for |
| --- | --- | --- | --- |
| `--ts-brand` | `#3f7a16` | `#8cc63f` | Switch on, current page (pure pagination), "today" ring |
| `--ts-brand-deep` | `#24541c` | `#a9d879` | Primary button, links, active nav icon, current page number |
| `--ts-brand-deep-hover` | `#3e6837` | `#bde395` | Primary button hover |
| `--ts-accent` | `#8cc63f` | `#8cc63f` | Nav marker bar, tab underline, focus ring, progress, accent button |
| `--ts-accent-hover` | `#7cb531` | `#9ed255` | Accent button hover |
| `--ts-accent-soft` | `#dceebb` | `#23301f` | Current child nav row, selected table row, "brand" icon chip |
| `--ts-accent-ink` | `#152109` | `#c2e595` | Text and icons on accent-soft |
| `--ts-ink` | `#1a2418` | `#2c2b43` | Open nav group, selected pill tab, toast, tooltip, save bar |
| `--ts-on-brand` / `--ts-on-accent` | `#fff` / `#152109` | `#0f1a0a` / `#152109` | Text on brand / accent fills |
| `--ts-on-ink-muted` | `#b5c5b2` | | Secondary text on ink (the open group's chevron) |
| `--ts-on-ink-chip` | `#ffffff29` | | A count pill on ink |
| `--ts-on-deep-disc` | `#ffffff2e` | | The round icon disc inside a primary header button |

Gradients for hero areas: `--ts-hero-green`, `--ts-hero-red`, `--ts-hero-teal`, `--ts-hero-charcoal`,
`--ts-ink-gradient` and `--ts-auth-gradient`, all at 150 to 155°. `--ts-hatch` is a 6px diagonal stripe for "no data"
or a disabled area.

### 3.3 Colour: status

Each tone has five values. Light values are shown here; dark ones are in the file.

| Tone | solid | fg | bg | border | chip |
| --- | --- | --- | --- | --- | --- |
| danger | `#a72e18` | `#8c2413` | `#fdf1ee` | `#f3cdc3` | `#fbe2dc` |
| warning | `#b07a16` | `#8a5f10` | `#fef8ec` | `#eedcb4` | `#fdf0da` |
| success | `#3f7a16` | `#2c5a10` | `#f4faea` | `#d8e9be` | `#edf7de` |
| info | `#2d7ff9` | `#1f5fbf` | `#f2f7fe` | `#cfe0fa` | `#e4eefb` |
| neutral | | `#48524a` | | | `#f1f3ed` |
| violet | | `#6b3fa0` | | | `#f1e7fb` |

- **solid**: a solid badge or the danger button.
- **fg**: text on any tint.
- **bg**: alert background.
- **border**: alert and caps-badge border.
- **chip**: badge, icon chip and change pill background.
- `--ts-danger-deep` (`#8c2413`) is the danger button's hover.

### 3.4 Colour: charts

Use them in this order: `--ts-chart-1` … `--ts-chart-8`.

- Light: `#3f7a16 #1f5fbf #9a6510 #6b3fa0 #147a74 #a72e18 #b8407a #5d6b5f`
- Dark: `#8cc63f #6aa2f2 #e0a93e #b48be6 #3fbfb3 #ee7a62 #e57fb0 #a2a1b8`

### 3.5 Radius

| Token | px | Used for |
| --- | --- | --- |
| `--ts-radius-pill` | 99 | Buttons, badges, chips, tabs, search, avatars, page buttons, switches |
| `--ts-radius-xl` | 30 | The app canvas |
| `--ts-radius-lg` | 24 | Dashboard link cards, tile buttons |
| `--ts-radius-md` / `--ts-radius-card` | 22 | Cards, panels, stat tiles, alerts, segmented-control track |
| `--ts-radius-modal` / `--ts-radius-drawer` / `--ts-radius-auth` | 26 | Modal, drawer, sign-in card |
| `--ts-radius-20` | 20 | Save bar |
| `--ts-radius-18` | 18 | (spare) |
| `--ts-radius-16` | 16 | Facts, view tiles and rows, bulk-selection bar |
| `--ts-radius-sm` / `--ts-radius-nav` | 14 | Inputs, selects, menus, popovers, table header ends, icon chips, nav rows |
| `--ts-radius-12` | 12 | Brand mark |
| `--ts-radius-10` | 10 | Child nav rows |
| `--ts-radius-xs` | 8 | Menu items, tooltips, kbd, pure-pagination buttons |

### 3.6 Spacing and sizes

| Token | px | | Token | px |
| --- | --- | --- | --- | --- |
| `--ts-space-1` | 4 | | `--ts-page-inset` | 16 |
| `--ts-space-2` | 8 | | `--ts-sidebar-width` | 246 |
| `--ts-space-3` | 10 | | `--ts-sidebar-padding-block` | 20 |
| `--ts-space-4` | 12 | | `--ts-card-padding` | 20 |
| `--ts-space-5` | 14 | | `--ts-modal-padding` | 24 |
| `--ts-space-6` | 18 | | `--ts-modal-width-sm / md / lg / xl` | 400 / 520 / 720 / 1200 |
| `--ts-space-7` | 22 | | `--ts-popover-width` | 320 |
| `--ts-space-8` | 26 | | `--ts-drawer-width` | 450 |
| `--ts-space-9` | 32 | | `--ts-toast-padding-block / inline` | 13 / 20 |

### 3.7 Control heights

| Token | px | Used for |
| --- | --- | --- |
| `--ts-control-xs` | 30 | Colour-mode segments, save-bar icon |
| `--ts-control-sm` | 34 | Small buttons, inputs, page buttons, segments, the avatar `sm` |
| `--ts-control` | 42 | Default buttons, inputs, underline tabs, avatar `md` |
| `--ts-control-lg` | 46 | Header buttons, icon chips, top-bar search and round buttons, avatar `lg` |
| `--ts-control-tab` | 38 | Pill tabs |
| `--ts-nav-row` | 44 | Sidebar rows |
| `--ts-nav-row-child` | 36 | Sidebar child rows |

### 3.8 Shadows, layers, focus

| Token | Value (light) | Used for |
| --- | --- | --- |
| `--ts-shadow-card` | `0 1px 2px #1a24180a` | Cards, panels, menus |
| `--ts-shadow-nav-active` | `0 2px 8px #1a24180f` | The current nav row |
| `--ts-shadow-segment-active` | `0 2px 6px #1a241814` | The checked segment, colour-mode segment |
| `--ts-shadow-modal` | `0 24px 70px #1a24184d` | Modal |
| `--ts-shadow-drawer` | `0 20px 60px #1a24183d` | Drawer |
| `--ts-shadow-toast-v3` | `0 16px 40px #1a241847` | Toast |
| `--ts-z-modal-backdrop` … `--ts-z-toast` | 1100, 1110 (modal), 1120 (menu), 1130 (tooltip), 1140 (toast) | Overlays |
| `--ts-focus-ring` | `2px solid var(--ts-accent)` | Outline on `:focus-visible` |
| `--ts-focus-offset` | `2px` | Outline offset |
| `--ts-focus-contour` | `0 0 0 6px var(--ts-brand-deep)` | A box-shadow outside the ring, so it shows on any background |

## 4. Typography

| Role | Font | Size | Weight | Tracking | Line height |
| --- | --- | --- | --- | --- | --- |
| Page title (`.ts-page-header__title`) | Outfit | 38 (`--ts-text-5xl`) | 600 | -.025em | 1.05 |
| h1 (base) | Outfit | 32 | 600 | -.025em | |
| Big figure (`.ts-stat-tile__figure`) | Outfit | 32, tabular numbers | 600 | -.025em | 1.1 |
| h2 | Outfit | 26 | 600 | -.025em | |
| Modal title | Outfit | 24 | 600 | | 1.2 |
| h3 | Outfit | 21 | 600 | | |
| Brand name in the sidebar | Outfit | 20 | 600 | -.01em | 1.2 |
| Card title (`.ts-view__card-title`) | Outfit | 19 | 600 | | 1.3 |
| Panel title (`.ts-panel__title`) | Plus Jakarta Sans | 18 | 600 | | |
| Sub-title (`.ts-view__sub-title`) | Outfit | 15 | 600 | | 1.35 |
| Body, reading (`.ts-view__body`) | Plus Jakarta Sans | 15 | 400 | | 1.65, max 78ch |
| Nav row, input, button md | Plus Jakarta Sans | 14 | 500 / 600 | | 1.4 |
| Body, UI (base) | Plus Jakarta Sans | 13.5 | 400 | | 1.5 |
| Small text, hints, badges | Plus Jakarta Sans | 12.5 / 12 | 400 / 500 | | 1.4 |
| Caps label (stat label, table head) | Plus Jakarta Sans | 11.5 | 700 | .12em (stat) / .06em (table) | 1.4 |
| Sidebar group label | Plus Jakarta Sans | 11 | 600 | .12em | |

**Rules**

- Outfit is only for titles and numbers. Plus Jakarta Sans is for all running text and controls.
- Numbers in tiles and tables use `font-variant-numeric: tabular-nums` (`.ts-numeric`).
- Keep subtitles and empty-state text to 70ch at most.
- Page titles are sentence case. Labels in caps are short: one to three words.

## 5. Layout: the frame

The frame is rebuilt from the screenshots.

```
┌──────────── ground ───────────────────────────────────────────────────────────────┐
│ ┌ sidebar 246 ┐  ┌ top bar ───────────────────────────────────────────────────────┐ │
│ │ [■] Brand    │  │ (☰) ( 🔍 Search…            ⌘K )        (◐) (🔔•) (◯ Name ▾)  │ │
│ │ MENU         │  └─────────────────────────────────────────────────────────────────┘ │
│ │ ▌⌂ Dashboard │  ┌ canvas: surface-sunken, radius 30, padding 32 ──────────────────┐ │
│ │  ▤ Bookings ▾│  │ Breadcrumb / Crumb                                               │ │
│ │  │ All       │  │ Page title 38px                           [Secondary] [Primary]  │ │
│ │  │ Calendar  │  │ One line of subtitle                                             │ │
│ │ GENERAL      │  │ [stat tile] [stat tile] [stat tile] [stat tile]                  │ │
│ │  ⚙ Settings  │  │ [panel ………………………………………………… 2fr ] [ panel …… 1fr ]                │ │
│ └──────────────┘  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

```html
<div class="ts-app" data-drawer="closed">
  <aside class="ts-app__sidebar"> … sidebar (6.1) … <p class="ts-app__sidebar-foot">…</p></aside>
  <div class="ts-app__scrim"></div>
  <div class="ts-app__main">
    <header class="ts-topbar">
      <button class="ts-icon-button ts-topbar__menu" data-size="lg" aria-label="Open menu">☰</button>
      <button class="ts-topbar__search"><svg…/><span>Search…</span><kbd class="ts-kbd">⌘K</kbd></button>
      <div class="ts-topbar__end">
        <button class="ts-icon-button" data-size="lg" aria-label="Theme">◐</button>
        <button class="ts-icon-button ts-topbar__dot" data-size="lg" aria-label="Alerts">🔔</button>
        <button class="ts-user-chip"><span class="ts-avatar" data-size="sm"><span class="ts-avatar__initials">SM</span></span>
          <span class="ts-user-chip__text"><b class="ts-user-chip__name">Name</b><span class="ts-user-chip__role">Role</span></span></button>
      </div>
    </header>
    <main class="ts-app__canvas"> page header, then sections </main>
  </div>
</div>
```

- **Sidebar.** It sits on the ground, is 246px wide, sticky, and scrolls on its own. It holds the brand, then group
  labels (MENU, GENERAL) with rows.
- **Top bar.**
  - The search is a 46px pill, up to 420px wide, with a `⌘K` key hint.
  - On the right are round 46px icon buttons on white, then the user chip.
- **Canvas.** It is the sunken surface with a 30px radius and 32px padding (22 / 14 on phones). It holds the page.
- **Below 64rem.**
  - The sidebar becomes a drawer: fixed, with a 26px radius on its open side and the drawer shadow over a scrim.
  - Set `data-drawer="open"` on `.ts-app` to show it, and close it on scrim click or Esc.
  - The menu button appears in the top bar.
- **Below 40rem.** The search shrinks to a round button, the user chip to its avatar, and the page title to 32px.
- **Page widths.** Content grids use container queries, not the viewport:
  - **View:** `2fr / 1fr` from a 1000px container width.
  - **Detail page:** `1.55fr / 1fr` from 860px.
  - **Form page:** a section nav beside the form from 960px.

## 6. Components

Each entry gives what the component is for, its markup, its sizes and states, and the key values. The full rules are in
`ts-components.css` under a heading with the component's name.

### 6.1 Sidebar (`.ts-sidebar__*`)

Navigation in groups. Rows are 44px tall with a 14px radius, `--ts-text-secondary`, 14px medium, and an 18px icon in
`--ts-text-faint`.

```html
<a class="ts-sidebar__brand" href="/"><img class="ts-sidebar__brand-mark" src="logo.svg" alt=""><span class="ts-sidebar__brand-name">DCLab</span></a>
<nav class="ts-sidebar__nav" aria-label="Main">
  <div class="ts-sidebar__nav-section">
    <span class="ts-sidebar__nav-group-label">Menu</span>
    <ul class="ts-sidebar__nav-list">
      <li class="ts-sidebar__nav-direct-item">
        <a class="ts-sidebar__nav-row ts-sidebar__nav-row-active" aria-current="page" href="/">
          <svg class="ts-sidebar__nav-row-icon"…/><span class="ts-sidebar__nav-row-label">Dashboard</span></a></li>
      <li class="ts-sidebar__nav-group-item">
        <button class="ts-sidebar__nav-row ts-sidebar__nav-row-group" aria-expanded="true">
          <svg class="ts-sidebar__nav-row-icon"…/><span class="ts-sidebar__nav-row-label">Runs</span>
          <svg class="ts-sidebar__nav-chevron ts-sidebar__nav-chevron-open"…/></button>
        <ul class="ts-sidebar__nav-children">
          <li class="ts-sidebar__nav-child-item"><a class="ts-sidebar__nav-row ts-sidebar__nav-row-child ts-sidebar__nav-row-current" aria-current="page" href="#">All runs</a></li>
        </ul></li>
    </ul>
  </div>
</nav>
```

| State | Look |
| --- | --- |
| Rest | Transparent; text-secondary; icon text-faint |
| Hover | `--ts-surface-hover` fill, text colour, icon inherits |
| Current (top level), `-active` or `-current` | White surface, `--ts-shadow-nav-active`, semibold, icon `--ts-brand-deep`, and a 4×26px accent pill 12px to the left of the row |
| Group open, `[aria-expanded=true]` | Ink fill, white text, icon in accent, chevron `--ts-on-ink-muted`, no pill |
| Child rest | 36px, radius 10, 13px, no icon |
| Child current | `--ts-accent-soft` fill, bold |
| Children list | 1px `--ts-border-strong` line on the left, indented 20px |
| Group label | 11px semibold uppercase, `.12em` tracking, text-secondary |

- **Brand.** The mark is 38px with a 12px radius. The name is in Outfit 20 semibold. It sits 22px above the nav.
- **Chevron.** It points right when the group is closed and down when open (add `-chevron-open`), turning in .15s.

### 6.2 Page header (`.ts-page-header`)

The top of every page: breadcrumbs, a big title, one line under it, and actions on the right, aligned to the title's
bottom.

```html
<header class="ts-page-header">
  <nav class="ts-page-header__breadcrumbs"><ol class="ts-page-header__crumbs">
    <li class="ts-page-header__crumb"><a class="ts-page-header__crumb-link" href="#">Runs</a></li>
    <li class="ts-page-header__crumb"><a class="ts-page-header__crumb-link ts-page-header__current" aria-current="page">Run 42</a></li></ol></nav>
  <div class="ts-page-header__lead">
    <h1 class="ts-page-header__title">Bookings</h1>
    <p class="ts-page-header__subtitle">Every booking across products.</p></div>
  <div class="ts-page-header__actions">
    <button class="ts-button" data-variant="secondary" data-size="md">Export</button>
    <button class="ts-button" data-variant="primary" data-size="md"><span class="ts-button__content"><span class="ts-button__icon" data-slot="icon">+</span><span class="ts-button__label">New booking</span></span></button></div>
</header>
```

- **Title:** Outfit 38/1.05, semibold, -.025em.
- **Subtitle:** 14px, text-secondary, max 70ch, 8px below the title.
- **Header buttons** are always 46px tall with 22px side padding:
  - a secondary button gets a 1.5px border;
  - a primary button's icon sits in a 22px white-alpha disc.
- **Breadcrumbs:** 12.5px, separated by a faint `/`; the current crumb is in text colour.

### 6.3 Button (`.ts-button`)

```html
<button class="ts-button" data-variant="primary" data-size="md">Save</button>
<a class="ts-button" data-variant="secondary" data-size="sm" href="#">Open</a>
<button class="ts-button ts-button__icon-only" data-variant="ghost" data-size="md" aria-label="More">…</button>
<button class="ts-button" data-variant="primary" data-size="md" data-state="loading">
  <span class="ts-button__content"><span class="ts-button__label">Saving</span></span>
  <span class="ts-button__spinner"><span class="ts-loader"><span class="ts-loader__ring"></span></span></span></button>
```

| Variant | Fill | Text | Hover |
| --- | --- | --- | --- |
| `primary` | `--ts-brand-deep` | `--ts-on-brand` | `--ts-brand-deep-hover` |
| `accent` | `--ts-accent` | `--ts-on-accent` | `--ts-accent-hover` |
| `secondary` | `--ts-surface` + 1px `--ts-border-strong` | text | `--ts-surface-hover` |
| `danger` | `--ts-danger-solid` | white | `--ts-danger-deep` |
| `ghost` | transparent | text-secondary | `--ts-surface-hover`, text |

- **Sizes** (height / side padding / font):
  - `sm`: 34 / 14 / 12.5
  - `md`: 42 / 18 / 14
  - `lg`: 46 / 22 / 14
- **Shape and weight:** a pill, semibold. Icon and label are 8px apart.
- **Disabled** (`disabled` or `aria-disabled="true"`): opacity .55; the hover does not change.
- **Loading** (`data-state="loading"`): the content is hidden and a spinner shows.
- **Use:**
  - one primary button per view;
  - secondary for the rest;
  - ghost inside dense rows;
  - accent only for a single highlighted call to action (for example "Upgrade");
  - danger only to confirm a destructive step.

### 6.4 Icon button (`.ts-icon-button`) and text button (`.ts-text-button`)

- **Icon button.** A round, transparent button with text-secondary colour. On hover it gets a surface-hover fill and
  text colour.
  - Sizes: `sm` 34, `md` 42, `lg` 46.
  - Always give it an `aria-label`.
  - In the top bar it gets a white fill and a hairline border.
- **Text button.** A pill with no fill and medium weight that underlines on hover, for low-key actions such as
  "Clear all".
- **Link** (`.ts-link`): brand-deep, semibold, 12.5px, underlined on hover.

### 6.5 Badge (`.ts-badge`)

```html
<span class="ts-badge" data-tone="success">Paid</span>
<span class="ts-badge" data-tone="warning" data-caps>Pending</span>
<span class="ts-badge" data-tone="danger" data-solid>Overdue</span>
```

- **Shape.** A pill, 12px medium, with 2px × 8px padding.
- **Tones:** `success`, `info`, `warning`, `danger`, `neutral`, `violet`, each using the chip fill and fg text.
- **Caps** (`data-caps`): 11.5px bold uppercase with `.12em` tracking, 4 × 10 padding, and the tone's border. Use it for
  a record's status.
- **Solid** (`data-solid`): the solid fill with white text. It exists for `success` and `danger` only, for the one
  status that must stand out.

### 6.6 Chip (`.ts-chip`)

An active filter or a tag that can be removed. It is a pill with a strong border, the muted fill and 12.5px text, and a
26px round remove button.

```html
<span class="ts-chip"><span class="ts-chip__label">Status: Paid</span><button class="ts-chip__remove-button" aria-label="Remove">×</button></span>
```

### 6.7 Stat tile (`.ts-stat-tile`)

A KPI: a caps label, a big number, an icon chip, and how it changed.

```html
<article class="ts-stat-tile">
  <div class="ts-stat-tile__head"><div class="ts-stat-tile__lead">
    <p class="ts-stat-tile__label">Bookings today</p><p class="ts-stat-tile__figure">128</p></div>
    <span class="ts-stat-tile__icon-chip" data-icon-tone="brand"><svg…/></span></div>
  <p class="ts-stat-tile__change" data-tone="positive"><span class="ts-stat-tile__change-value">▲ 12%</span><span class="ts-stat-tile__change-caption">vs last week</span></p>
</article>
```

- **Card:** 20px padding, 22px radius, hairline border, white, no shadow.
- **Label:** 11.5px bold caps, text-secondary.
- **Figure:** Outfit 32, semibold, tabular numbers, 18px under the label.
- **Icon chip:** 46px, 14px radius. Tones: `brand` (accent-soft with brand-deep), `success`, `danger`, `warning`,
  `info`, `violet`.
- **Change pill:** 11.5px bold. `positive` is the success chip, `negative` the danger chip, and with no tone it is muted.
  The caption is 12.5px secondary.
- **Lay tiles out** in `.ts-stat-grid` (auto-fit, 13rem minimum, 12px gap). Four is ideal.

### 6.8 Panel (`.ts-panel`) and view card (`.ts-view__card`)

**Panel.** The general container: 20px padding, 22px radius, hairline border, the card shadow.

- Header: a title (18px semibold) with a subtitle (12.5 secondary) on the left and actions on the right.
- The body starts 14px below the header.

```html
<section class="ts-panel">
  <div class="ts-panel__header"><div class="ts-panel__heading"><h2 class="ts-panel__title">Recent runs</h2><p class="ts-panel__subtitle">Last 7 days</p></div>
    <div class="ts-panel__actions"><button class="ts-button" data-variant="ghost" data-size="sm">View all</button></div></div>
  <div class="ts-panel__body">…</div>
</section>
```

**View card.** The card of a detail view. Its title is in Outfit 19, and its parts sit in a 14px-gap grid.

- Inside it:
  - `.ts-view__label` is a caps label;
  - `.ts-view__text` is 13px secondary;
  - `.ts-view__body` is 15/1.65 reading text;
  - `.ts-view__tiles` / `.ts-view__tile` are small facts on the input fill with a 16px radius;
  - `.ts-view__rows` / `.ts-view__row` are list rows on the input fill;
  - `.ts-view__field-box` is a read-only value;
  - `.ts-view__divided` adds a top rule.
- `.ts-view__layout` is the page grid: one column, then `2fr 1fr` (`.ts-view__main` + `.ts-view__rail`) at a 1000px
  container width.

### 6.9 Tabs (`.ts-tabs`)

There are two looks. **Underline** is for the sections of a page. **Pill** is for switching what a list shows.

```html
<div class="ts-tabs">
  <div class="ts-tabs__list" data-variant="underline" role="tablist">
    <button class="ts-tabs__tab" role="tab" aria-selected="true">Overview</button>
    <button class="ts-tabs__tab" role="tab" aria-selected="false">Columns <span class="ts-tabs__count">12</span></button>
  </div>
  <div class="ts-tabs__panel" role="tabpanel">…</div>
</div>
```

- **Underline.**
  - Tabs are 42px tall with 14px side padding, 13.5px medium, text-secondary.
  - There is a 2.5px transparent bottom border, and the list has a 1px hairline under it.
  - Selected: the accent underline, text colour, bold.
  - When the list overflows, fading edges show (gradients with `background-attachment: local`).
- **Pill** (`data-variant="pill"`).
  - The track is a sunken pill with 4px padding.
  - Tabs are 38px with 18px padding, bold.
  - Selected: ink fill with white text; the count pill becomes `--ts-on-ink-chip`.
- **Count:** a small pill, hover fill, 11.5px bold.
- **Vertical** (`data-orientation="vertical"` on `.ts-tabs`): the list sits beside the panel and the marker moves to the
  left edge.

### 6.10 Segmented control (`.ts-segmented-control__*`)

A choice among two to five options that changes a view in place (for example Day / Week / Month). It is built from
radio inputs.

```html
<div class="ts-segmented-control__group" role="radiogroup" aria-label="Range">
  <label class="ts-segmented-control__segment"><input class="ts-segmented-control__input" type="radio" name="r" checked>Week</label>
  <label class="ts-segmented-control__segment"><input class="ts-segmented-control__input" type="radio" name="r">Month <span class="ts-segmented-control__count">3</span></label>
</div>
```

- **Track:** the muted surface, a 22px radius, 4px padding.
- **Segment:** a pill at least 34px tall with 14px padding, 12.5px semibold, text-secondary.
- **Checked:** a white fill with the segment shadow.

### 6.11 Data table (`.ts-data-table`)

```html
<div class="ts-data-table"><div class="ts-data-table__frame"><div class="ts-data-table__scroller">
  <table class="ts-data-table__table">
    <thead><tr>
      <th class="ts-data-table__header-cell" aria-sort="descending"><button class="ts-data-table__sort-button">Date <svg class="ts-data-table__sort-icon"…/></button></th>
      <th class="ts-data-table__header-cell">Product</th>
      <th class="ts-data-table__header-cell ts-data-table__header-actions"><span class="sr-only">Actions</span></th></tr></thead>
    <tbody class="ts-data-table__body">
      <tr><td class="ts-data-table__cell">19 Oct</td><td class="ts-data-table__cell">…</td><td class="ts-data-table__cell ts-data-table__cell-actions">…</td></tr>
      <tr data-selected><td class="ts-data-table__cell">…</td>…</tr>
    </tbody></table></div></div></div>
```

- **Frame:** white, with a 14px radius that clips the table.
- **Header cells:**
  - 12px / 14px padding, on the sunken fill;
  - 11.5px bold uppercase, `.06em` tracking, text-secondary;
  - sticky at the top;
  - the first and last have rounded outer corners (14px), so the header reads as a rounded bar.
- **Cells:** 12 / 14 padding, 13.5px, a 1px hairline under each row.
  - Row hover: `--ts-surface-row-hover`.
  - Selected row (`data-selected`): `--ts-accent-soft`.
- **Sort:** the icon is faint until the column is sorted (`aria-sort`), then takes the text colour.
- **Row actions** (`.ts-row-actions`): inline ghost buttons, plus a menu (`.ts-data-table__menu`, items
  `.ts-data-table__menu-item`, `data-tone="danger"` for delete).
- **Empty or error:** `.ts-data-table__notice-cell` with an empty state inside.
- **Footer** (`.ts-table-footer__*`):
  - a selection status on the left;
  - "Page 2 of 9" and 34px pill page buttons on the right;
  - the current page (`aria-current="page"`) is filled brand-deep;
  - `.ts-table-footer__page-gap` is the "…".
- **Pure pagination** (`.ts-pure-pagination__*`): an older look with 36px buttons and an 8px radius; the active page is
  brand-filled.

### 6.12 Inputs: text, search, select, textarea

All fields share one anatomy and the same three sizes.

```html
<div class="ts-input__field" data-size="md" data-state="rest">
  <label class="ts-input__label" for="email">Email</label>
  <p class="ts-input__hint">We never share it.</p>
  <div class="ts-input__control ts-input__control-leading">
    <span class="ts-input__adornment"><svg…/></span>
    <input class="ts-input" id="email" placeholder="name@company.com"></div>
  <div class="ts-input__meta"><p class="ts-input__error"><svg class="ts-input__error-icon"…/>Enter a valid email</p></div>
</div>
```

- **Box:**
  - a 1px `--ts-border-strong` border, a 14px radius, the input fill;
  - 14px text (12.5 for `sm`);
  - heights 34 / 42 / 46 with 14 / 18 / 22 side padding.
- **Label:** bold, 12.5 (12 for `sm`), 8px above the box.
- **Hint:** 12.5px, text-secondary.
- **States** (on the field):
  - `data-state="rest"`: hover fill;
  - `readonly`: muted fill, default cursor;
  - `disabled`: opacity .55 and a not-allowed cursor;
  - `data-invalid`: a danger-fg border and an error line in danger-fg with an icon.
- **Adornments:** a leading icon sits 12px in, and the text is padded past it. A trailing control (for example "show")
  sits at the end.
- **Search** (`.ts-search-field__field` / `.ts-search-field__input`): the same, with a leading magnifier and a round
  clear button.
- **Select** (`.ts-select__field` / `.ts-select`): the same box, with `appearance: none` and a caret 12px from the end.
- **Textarea** (`.ts-textarea__field` / `.ts-textarea`): the same box, 12px top and bottom padding, resizes vertically.

### 6.13 Checkbox, radio, switch

Each control is a visually hidden native input next to a drawn box. The input keeps a 26px hit area.

- **Checkbox** (`.ts-checkbox__*`):
  - an 18px box (14 `sm`, 22 `lg`) with a radius of 3/16 of its size, a strong border and white fill;
  - checked or indeterminate: an accent fill with a brand border, and the glyph in `--ts-on-accent`.
- **Radio** (`.ts-radio__*`):
  - an 18px circle that is filled when checked;
  - groups use `.ts-radio__group` with a legend, and options are vertical or `data-orientation="horizontal"`.
- **Switch** (`.ts-switch__*`):
  - the track is 2× its height (44 × 22 at `md`) on the muted fill;
  - the thumb is text-muted at rest;
  - on: a brand track with an on-brand thumb that slides one track-height;
  - `data-tone="danger"` uses danger-solid when on.
- **Labels:** 14px medium (12.5 for `sm`). An optional description sits below in 12.5 secondary.

```html
<label class="ts-switch__checkable" data-size="md"><span class="ts-switch__row">
  <input class="ts-switch__input" type="checkbox" role="switch" checked>
  <span class="ts-switch__track"><span class="ts-switch__thumb"></span></span>
  <span class="ts-switch__label-text">Email alerts</span></span></label>
```

### 6.14 Alert (`.ts-alert`)

```html
<div class="ts-alert ts-alert__banner" data-tone="warning" role="status">
  <svg class="ts-alert__mark"…/>
  <div class="ts-alert__content">
    <div class="ts-alert__title-row"><p class="ts-alert__title">Sync is behind</p></div>
    <p class="ts-alert__body">The last sync was 3 hours ago.</p>
    <div class="ts-alert__actions"><button class="ts-button" data-variant="secondary" data-size="sm">Sync now</button></div></div>
  <button class="ts-alert__dismiss-button" aria-label="Dismiss">×</button>
</div>
```

- **Banner:** a 22px radius, 14 × 18 padding, the tone's bg and border, with fg text.
- **Title:** 15.5 semibold. **Body:** 12.5/1.5.
- **Tones:** `success`, `warning`, `danger`, and `info`, which is neutral (sunken fill, strong border), not blue.
- **Note** (`.ts-alert__note`): a compact one-line version with a 14px radius and 10 × 12 padding, for use inside forms.
- **Dismiss:** a 26px round button whose hover fill is the tone's chip.

### 6.15 Empty state (`.ts-empty-state`)

The empty state is centred, with 12px gaps:

- a 46px icon chip (muted, 14px radius);
- a title in 18px semibold;
- a body in 14px secondary, at most 70ch;
- the actions.

`data-tone="danger"` turns the chip and title red, for load errors.

### 6.16 Avatar (`.ts-avatar`)

A round avatar with a hairline border and the muted fill, showing initials in semibold or an image.

- Sizes: `sm` 34 (12.5px), `md` 42 (14px), `lg` 46 (15.5px).

### 6.17 Modal (`.ts-modal__*`)

- **Backdrop:** `--ts-scrim-modal`.
- **Panel:**
  - widths: `sm` 400, `md` 520, `lg` 720, `xl` 1200;
  - 24px padding, a 26px radius, the modal shadow;
  - flex column, at most the viewport height minus 44.
- **Title:** Outfit 24 semibold. The close button is a 42px icon button aligned to the title.
- **Body:** 13.5/1.5 secondary; it scrolls.
- **Footer:** buttons at the right, 20px above. Under 480px they stack full-width.
- `.ts-confirm-dialog__*` adds a body and a failure line for "Are you sure?" dialogs.

### 6.18 Toast (`.ts-toast`)

A dark pill (ink, white text, 13.5 medium) with 13 × 20 padding and the toast shadow.

- A coloured mark leads the toast: `--ts-toast-mark-success`, `-info`, `-warning` or `-danger`.
- An optional round dismiss button closes it.

### 6.19 Tooltip, popover, menu

- **Tooltip** (`.ts-tooltip`): ink, white 12.5px text, 8 × 10 padding, an 8px radius, at most 400px wide.
- **Info term** (`.ts-info-term__term` + `__mark`): a term with a faint ⓘ that opens a tooltip or popover.
- **Popover** (`.ts-popover__panel`): 320px wide, 20px padding, a 14px radius, a hairline border and the card shadow.
  The title is 14 semibold.
- **Menu** (`.ts-menu-button__*`): a panel at least 204px wide with 8px padding, a 14px radius and the card shadow.
  - Items: 10 × 12 padding, an 8px radius, 14px text, 10px gap to the icon.
  - A focused item gets the hover fill. `data-tone="danger"` items are red.
  - `.ts-menu-button__check-slot` reserves 18px for a check mark.

### 6.20 Loader (`.ts-loader`)

- **Ring:** 26px, 2px, two sides coloured, spinning in .75s. With reduced motion it pulses instead.
- **Skeleton** (`.ts-loader__skeleton`): a sweeping sheen.
- **Splash** (`.ts-loader__splash`): a full-screen ground with the logo (46px), a quote in Outfit 24, and a 3px progress
  bar in accent.

### 6.21 Filter bar, list page, save bar

- **Filter bar** (`.ts-filter-bar`):
  - controls wrap, each at least 252px;
  - below them, the active filter chips, "Clear all" and a result count on the right in 12.5 secondary;
  - `data-labels="hidden"` hides the field labels visually for a compact one-line bar.
- **List page** (`.ts-list-page`):
  - a toolbar (start: search and filters; end: actions);
  - an 18px-gap stack;
  - a bulk bar (sunken, 16px radius) showing "3 selected" and its actions;
  - a footer.
- **Save bar** (`.ts-save-bar`):
  - an ink bar with a 20px radius and 14 × 18 padding, saying "You have unsaved changes" with Discard / Save;
  - `data-sticky` keeps it at the bottom of the form.

### 6.22 Detail page and form page

- **Detail page** (`.ts-detail-page`):
  - Facts (`__facts`, `__fact`) are tiles on the sunken fill with a 16px radius, a 12px label and a value in
    Outfit 15.5.
  - Lists (`__list-row`) are label / value rows with 180px labels from a 480px container width.
  - Columns become `1.55fr / 1fr` with an aside (`data-has-aside`) from 860px.
- **Form page** (`.ts-form-page`):
  - sections with a heading (title and description) and a body of fields;
  - from 960px, a sticky section nav (`__nav-list`, `__nav-link`) sits beside them.

### 6.23 Dashboard link cards (`.ts-admin-dashboard__*`)

Cards that lead to a section, in a grid of three, auto-fill below 60rem.

- **Card:** a 24px radius, 12px padding, a hairline border.
- **Hover:** an accent border, the hover fill, and the title underlined.
- **Icon:** a 46px accent-soft square (22px radius) with accent-ink.
- **Title:** 14 semibold. **Description:** 12.5 secondary. Then a "View →" line in accent-ink.

### 6.24 Colour mode (`.ts-colour-mode__*`)

A caps label ("Theme") over three equal segments: Light, Dark, System. The track is muted with an 8px radius. The
pressed segment (`aria-pressed="true"`) is white with the segment shadow. `.ts-colour-mode__mark` is a half-filled
circle icon.

## 7. Page patterns

These recur on every TripsSource page. Build new pages from them.

1. **Overview / dashboard**:
   - the page header;
   - a stat grid of four tiles;
   - "Needs attention" cards (`.ts-attention`, with a coloured edge from `data-tone`, a caps label, a 28px count and
     one line);
   - a `2fr / 1fr` view layout: the main panel (a table or chart) and a rail of smaller panels.
2. **List**:
   - the page header with a primary "New" button;
   - underline or pill tabs for saved views;
   - a filter bar;
   - a data table with row actions;
   - the table footer.
3. **Detail**:
   - the page header with breadcrumbs and actions;
   - underline tabs;
   - a detail page with facts on top, then view cards, and an aside for status and people.
4. **Form**:
   - the page header;
   - a form page with sections and a section nav;
   - a sticky save bar.
5. **Empty and error**: an empty state centred in the panel where the content would be, never a blank page.

**Spacing between parts.**

- Page header → first section: 22px.
- Between sections: 18 to 22px.
- Inside a panel: 14px.
- Between tiles and cards: 12px.

## 8. States, focus and motion

- **Hover:** a fill change only (`--ts-surface-hover`, or the row-hover colour). Text darkens from secondary to text.
  Nothing moves, except pure-pagination buttons (`translateY(-1px)`).
- **Focus** (`:focus-visible` only):
  - a 2px accent outline at a 2px offset;
  - plus a 6px `--ts-focus-contour` shadow in the deep brand colour, so the ring reads on any fill;
  - inside lists, the inset contour is used instead.
- **Disabled:** opacity .55 and a not-allowed cursor. Never grey out with a different colour.
- **Current / selected:**
  - the nav row is white with a pill marker;
  - a child row and a table row get the accent-soft fill;
  - a tab gets the accent underline, or an ink pill;
  - a segment is white with a shadow;
  - a page number is brand-deep.
- **Invalid:** a danger-fg border, and an error line with an icon under the field.
- **Motion:**
  - .15s on colours, the chevron and the switch thumb; .2s on the drawer;
  - the loader spins at .75s;
  - `prefers-reduced-motion` turns spinners into pulses and stops the splash animations.
- **Hit areas:** checkboxes, radios and switches keep a 26px invisible hit area around the drawn control. Every icon
  button is at least 34px.

## 9. The blue and light-green theme

`ts-theme-blue.css` changes colours only and keeps every role:

| Role | TripsSource | Blue theme (light) | Blue theme (dark) |
| --- | --- | --- | --- |
| brand | `#3f7a16` | `#2563eb` | `#5b9bff` |
| brand-deep (primary button, links) | `#24541c` | `#1d4ed8` | `#8ab8ff` |
| accent (marker, underline, focus) | `#8cc63f` | `#8bd346` | `#a3e635` |
| accent-soft (current row) | `#dceebb` | `#e4f5d2` | `#1d2c19` |
| ink (open group, pill tab, toast) | `#1a2418` | `#0f1a2e` | `#26324d` |
| ground | `#e7e8e3` | `#e8ecf2` | `#0b1120` |
| text | `#1a2418` | `#0f1a2e` | `#e8edf7` |

The neutrals move from green-grey to cool slate. Status colours stay TripsSource's, so "success" is still green and
"info" still blue. Charts lead with blue, then green.

**Contrast (WCAG):**

- white on the primary button: 6.7:1;
- dark on the light-green accent: 8.5:1;
- secondary text on white: 7.9:1;
- the dark theme's primary button: 8.9:1.

To use your own pair, override only `--ts-brand`, `--ts-brand-deep`, `--ts-brand-deep-hover`, `--ts-accent`,
`--ts-accent-hover`, `--ts-accent-soft` and `--ts-accent-ink`, in light and dark.

## 10. Using it in React or Next.js

- Import the three CSS files once, in the root layout:

  ```ts
  import './ts-tokens.css';
  import './ts-theme-blue.css';
  import './ts-components.css';
  ```

- Load the fonts with `next/font/google`. Map them onto the tokens, because TripsSource itself loads them with
  `next/font` under the names "fontDisplay" and "fontBody":

  ```ts
  const display = Outfit({ subsets: ['latin'], weight: ['400', '500', '600', '700'], variable: '--ts-font-display' });
  const body = Plus_Jakarta_Sans({ subsets: ['latin'], weight: ['400', '500', '600', '700'], variable: '--ts-font-body' });
  // <html data-ts-reset className={`${display.variable} ${body.variable}`}>
  ```

- Wrap each component once, with props mapped to the attributes. For example:

  ```tsx
  export function Button({ variant = 'secondary', size = 'md', ...p }: { variant?: 'primary' | 'accent' | 'secondary' | 'danger' | 'ghost'; size?: 'sm' | 'md' | 'lg' } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
    return <button className="ts-button" data-variant={variant} data-size={size} {...p} />;
  }
  ```

- With Tailwind, keep the kit's classes for components, and map the tokens in `theme.extend` (for example
  `colors: { brand: 'var(--ts-brand)' }`) for one-off layout.
- Icons: TripsSource uses 18px line icons with a 1.9 stroke, round caps and `currentColor`. Lucide icons match at
  `size={18} strokeWidth={1.9}`.

## 11. Class reference

The components, in the order of `ts-components.css`.

| Component | Root and main parts |
| --- | --- |
| Base | `[data-ts-reset]`, `.ts-text-*`, `.ts-weight-*`, `.ts-tracking-*`, `.ts-label-caps`, `.ts-numeric`, `.ts-font-display`, `.ts-font-body` |
| AdminDashboard | `__page`, `__tiles`, `__tile-button`, `__links`, `__link-card`, `__link-head`, `__link-icon`, `__link-title`, `__link-description`, `__link-view`, `__toolbar`, `__search`, `__count` |
| Alert | `.ts-alert`, `__banner`, `__mark`, `__content`, `__title-row`, `__title`, `__body`, `__actions`, `__note`, `__note-text`, `__dismiss-button` |
| Avatar | `.ts-avatar`, `__image`, `__initials` |
| Badge | `.ts-badge` |
| Button | `.ts-button`, `__content`, `__icon`, `__label`, `__spinner`, `__icon-only`, `__full-width` |
| Checkbox | `__checkable`, `__row`, `__input`, `__box`, `__glyph`, `__glyph-check`, `__glyph-minus`, `__label-text`, `__description` |
| Chip | `.ts-chip`, `__label`, `__remove-button` |
| ColourMode | `__choice`, `__choice-label`, `__segments`, `__segment`, `__mark` |
| ConfirmDialog | `__body`, `__failure` |
| CrmStatusBadge | `__status`, `__derived`, `__none` |
| DataTable | `.ts-data-table`, `__frame`, `__scroller`, `__table`, `__header-cell`, `__sort-button`, `__sort-icon`, `__body`, `__cell`, `__cell-actions`, `__cell-selection`, `__menu`, `__menu-item`, `__notice-cell`, `__skeleton` |
| DateField | `.ts-date-field` and parts (calendar popover) |
| DetailPage | `.ts-detail-page`, `__columns`, `__main`, `__aside`, `__facts`, `__fact`, `__fact-label`, `__fact-value`, `__list`, `__list-row`, `__list-label`, `__list-value` |
| EmptyState | `.ts-empty-state`, `__glyph-chip`, `__title`, `__body`, `__action` |
| FileUpload | `.ts-file-upload` and parts |
| FilterBar | `.ts-filter-bar`, `__controls`, `__meta`, `__chips`, `__chip-item`, `__clear-all`, `__count` |
| FormPage | `.ts-form-page`, `__layout`, `__nav-list`, `__nav-link`, `__sections`, `__section`, `__section-header`, `__section-title`, `__section-description`, `__section-body`, `__fields` |
| Icon | `__default`, `__micro`, `__success-mark`, `__danger-mark` |
| IconButton | `.ts-icon-button` |
| InfoTerm | `__term`, `__mark` |
| Input | `__field`, `__label`, `__hint`, `__control`, `__control-leading`, `__control-trailing`, `.ts-input`, `__adornment`, `__trailing-control`, `__meta`, `__error`, `__count` |
| LegalLinksNav | `__legal` |
| Link | `.ts-link` |
| ListPage | `.ts-list-page`, `__toolbar`, `__toolbar-start`, `__toolbar-end`, `__stack`, `__bulk`, `__bulk-count`, `__bulk-actions`, `__footer`, `__meta` |
| Loader | `.ts-loader`, `__ring`, `__overlay`, `__skeleton`, `__splash`, `__stage`, `__mark`, `__quote`, `__progress`, `__progress-fill` |
| MenuButton | `.ts-menu-button`, `__menu`, `__item`, `__check-slot`, `__caret` |
| Modal | `__backdrop`, `__position`, `__panel`, `__title`, `__close-button`, `__body`, `__footer` |
| Overview | an operations table (rows, date cells, group rows, assign buttons). TripsSource-specific; use it as an example of a dense table |
| OverviewFilters | `__kpis`, `__kpi`, `__card`, `__row`, `__toggles`, `__search`, `__dates`, `__active` |
| PageHeader | `.ts-page-header`, `__breadcrumbs`, `__crumbs`, `__crumb`, `__crumb-link`, `__current`, `__lead`, `__title`, `__subtitle`, `__actions` |
| PageLoader | `__page` |
| Panel | `.ts-panel`, `__header`, `__heading`, `__title`, `__subtitle`, `__actions`, `__body` |
| Popover | `__panel`, `__title`, `__click-away` |
| PurePagination | `__pager`, `__button`, `__page`, `__nav`, `__active`, `__disabled`, `__ellipsis` |
| Radio | `__group`, `__legend`, `__options`, `__choice`, `__row`, `__input`, `__circle`, `__dot`, `__label-text`, `__error` |
| RowActions | `.ts-row-actions`, `__inline-action`, `__anchor` |
| RowDetails | `__details-cell`, `__details-list`, `__details-term`, `__details-value`, `__group-header` |
| SaveBar | `.ts-save-bar`, `__icon`, `__message`, `__actions` |
| SearchField | `__field`, `__label`, `__control`, `__input`, `__adornment`, `__clear-button`, `__meta`, `__error` |
| SearchableSelect | `.ts-searchable-select` and parts (combobox) |
| SegmentedControl | `__group`, `__segment`, `__input`, `__count` |
| Select | `__field`, `__label`, `__hint`, `__control`, `.ts-select`, `__caret`, `__error` |
| Sidebar | `__brand`, `__brand-mark`, `__brand-name`, `__nav`, `__nav-section`, `__nav-group-label`, `__nav-list`, `__nav-row`, `__nav-row-active`, `__nav-row-current`, `__nav-row-group`, `__nav-row-child`, `__nav-row-icon`, `__nav-row-label`, `__nav-chevron`, `__nav-chevron-open`, `__nav-children` |
| SkipLink | `.ts-skip-link` |
| StatTile | `.ts-stat-tile`, `__head`, `__lead`, `__label`, `__figure`, `__icon-chip`, `__change`, `__change-value`, `__change-glyph`, `__change-caption` |
| Summary | `__card`, `__name`, `__meta`, `__toggle`, `__promo` |
| Switch | `__checkable`, `__row`, `__input`, `__track`, `__thumb`, `__label-text`, `__description` |
| TableFooter | `.ts-table-footer`, `__selection-status`, `__pagination-nav`, `__page-status`, `__pages`, `__page-button`, `__page-number`, `__page-gap` |
| Tabs | `.ts-tabs`, `__list`, `__tab`, `__count`, `__panel` |
| TextButton | `.ts-text-button` |
| Textarea | `__field`, `__label`, `__control`, `.ts-textarea`, `__adornment`, `__meta` |
| Toast | `.ts-toast`, `__mark`, `__content`, `__dismiss-button` |
| Tooltip | `.ts-tooltip` |
| View | `.ts-view`, `__layout`, `__main`, `__rail`, `__card`, `__card-head`, `__heading`, `__card-title`, `__sub-title`, `__block`, `__label`, `__text`, `__body`, `__chips`, `__divided`, `__tiles`, `__tile`, `__tile-value`, `__field`, `__field-box`, `__rows`, `__row`, `__row-title`, `__row-sub` |
| Frame (rebuilt) | `.ts-app`, `__sidebar`, `__sidebar-foot`, `__scrim`, `__main`, `__canvas`; `.ts-topbar`, `__menu`, `__search`, `__end`, `__dot`; `.ts-kbd`; `.ts-user-chip`, `__name`, `__role`; `.ts-stat-grid`; `.ts-attention-grid`, `.ts-attention`, `__label`, `__count`, `__text` |
