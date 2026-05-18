# Demo Upgrade Roadmap

Goal: make the 3D artifact reconstruction pipeline visually inspectable in the browser.

## Current State

- GitHub Pages surface is live.
- README explains the point-cloud pipeline, synthetic demo mode, and training result snapshot.
- Training curve image is already visible.
- A deterministic browser fixture now lives at `docs/assets/demo_alignment.json`.
- A compact generated metrics report now lives at `examples/demo_alignment_report.json`.

## Highest-Impact Improvements

| Priority | Upgrade | Recruiter value |
| --- | --- | --- |
| Done | Add a browser viewer with scrambled/aligned/target fragment states. | Makes the project immediately visual. |
| Done | Add a tiny synthetic sample artifact under `docs/assets/` and `examples/`. | Lets reviewers reproduce without huge scans. |
| Done | Add alignment metrics and transform values beside the viewer/report. | Shows engineering depth beyond visuals. |
| P1 | Add a GIF of the reconstruction pipeline. | Improves README and profile scan value. |
| P2 | Add a notebook that runs on the small sample artifact. | Makes the pipeline reproducible. |

## Suggested Demo Shape

- Static GitHub Pages viewer using preconverted JSON geometry.
- Controls: scrambled fragments, aligned fragments, target geometry, metric overlay.
- Next data upgrade: export model-produced alignment outputs from real PLY fragments into the same JSON schema.

## Definition Of Done

- Reviewer can open one URL and see fragment alignment before/after/target states.
- Demo includes a small generated dataset and a short note on how the full pipeline scales.
- README links directly to the visual demo and result image.
