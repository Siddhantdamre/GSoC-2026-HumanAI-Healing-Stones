# Demo Fixtures

This directory contains small generated artifacts for review and testing.

The full Healing Stones pipeline targets large PLY scans. These fixtures keep a
tiny deterministic alignment case in the repository so mentors and reviewers can
understand the completion path without downloading full artifact data.

Regenerate the browser fixture and metrics report with:

```bash
python src/generate_demo_alignment.py
```

Generated outputs:

- `docs/assets/demo_alignment.json` - point data consumed by the GitHub Pages viewer.
- `examples/demo_alignment_report.json` - compact metrics and transform report.
