# CS 180 project site

## Division of labor — read this first

**Claude helps with design and code only. All written content comes from Samarth.**

Do not write, draft, or "fill in" any of the following, even as a suggestion or an
example, and even when a section is obviously empty:

- Body copy, overviews, explanations, analysis, or conclusions
- Figure captions and alt text describing what a photo shows
- Project titles, subtitles, blurbs, and taglines
- Interpretations of what a photograph demonstrates
- Measured values (focal lengths, distances, frame counts) — these are findings, and
  they are his to report, even when they can be read out of a file's EXIF

Where content is missing, leave a **placeholder**: short, generic, marked with
`class="placeholder"` so it renders in grey italic and is visibly unwritten.
Examples: `Overview.`, `Caption.`, `Your analysis.`, `— mm`. Never a plausible-sounding
draft that could be mistaken for finished text.

Claude *is* responsible for: layout, typography, color, spacing, CSS, JavaScript,
responsive behavior, accessibility, image processing (resizing, cropping, GIF assembly),
and wiring assets into the page.

If a design decision depends on what the content will say, ask rather than invent it.

## Structure

- `index.html` — home page; a directory of projects
- `proj0.html` — Project 0 writeup (one page per project)
- `style.css` — shared stylesheet
- `assets/projNN/` — original photographs, as delivered by Samarth
- `assets/projNN/web/` — generated web-sized derivatives; never edit by hand, and never
  modify anything in the original folders

## Design language

Copied from `~/eyerobot2.github.io`, Samarth's own project page. Mirror its tokens rather
than inventing new ones:

- Open Sans body, 16px / 1.6, `#333` on white, `padding: 2em`
- `'Avenir Next', Avenir` for all headings
- `--content-width: 800px`; `.section` is centered at that width
- Centered title block: `.title` 54px weight 700, `.subheader` beneath it
- Fixed 150px `.section-index` left rail — 11px, `#aaa` / `#929292` / active `#292929`,
  `border-right: 1px solid #e2e2e2`, hidden below 1180px
- 10px border-radius on all media
- Accent `#488af3`, used sparingly

Beyond the reference: `.compare` (drag-to-compare slider), `.exif` (capture-data strips),
`.data` (results table), `.placeholder`.

## House rules

- Keep it responsive to mobile, keyboard-focusable, and respectful of reduced motion
- Placeholders use `.frame--empty` / `.layer--empty` for images, `.placeholder` for text
- Don't add analytics, trackers, or third-party scripts
