# Demo Upgrade Roadmap

Goal: make the 3D artifact reconstruction pipeline visually inspectable in the browser.

## Current State

- GitHub Pages surface is live.
- README explains the point-cloud pipeline and training result snapshot.
- Training curve image is already visible.

## Highest-Impact Improvements

| Priority | Upgrade | Recruiter value |
| --- | --- | --- |
| P0 | Add a small Three.js viewer with before/after fragment alignment. | Makes the project immediately visual. |
| P0 | Add a tiny synthetic sample artifact under `examples/`. | Lets reviewers reproduce without huge scans. |
| P1 | Add alignment metrics and transform values beside the viewer. | Shows engineering depth beyond visuals. |
| P1 | Add a GIF of the reconstruction pipeline. | Improves README and profile scan value. |
| P2 | Add a notebook that runs on the small sample artifact. | Makes the pipeline reproducible. |

## Suggested Demo Shape

- Static GitHub Pages viewer using small `.ply`, `.obj`, or preconverted JSON geometry.
- Controls: original fragments, predicted transform, refined alignment, metric overlay.
- Fallback: screenshots if browser 3D assets are too large.

## Definition Of Done

- Reviewer can open one URL and see fragment alignment before/after.
- Demo includes a small dataset and a short note on how the full pipeline scales.
- README links directly to the visual demo and result image.
