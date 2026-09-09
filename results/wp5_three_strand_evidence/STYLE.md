# Publication style and caption contract

Use the Okabe-Ito palette in `scripts/publication_style.py`, DejaVu Sans, 8 pt body text, 9 pt axis labels, and 10 pt titles at final physical size. Export SVG and PDF plus a 300-dpi PNG preview from one figure object. Use marker, line-style, or hatch redundancy in addition to color. Never connect missing observations or render them as zero.

Every quantitative caption states the population and split, observational-unit count, primary metric and direction, estimate, uncertainty construction, seeds or repeats, compute boundary, evidence status, and interpretation limit.

The cross-study conceptual figure uses cumulative ToMe as the main token-merging illustration. Temporary `spatial_block` merging is supplemental: it merges before attention and restores the grid after each block. It is distinct from cumulative final-token reduction and must not be labeled final-K16.
