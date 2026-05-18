"""Generate a lightweight alignment fixture for the GitHub Pages demo.

The full project works with large PLY scans and Open3D. This script creates a
small deterministic JSON fixture so reviewers can inspect the reconstruction
idea in a browser without downloading artifact scans or installing native
3D dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "demo_alignment.json"
DEFAULT_REPORT = ROOT / "examples" / "demo_alignment_report.json"


def _round_point(point: tuple[float, float, float]) -> list[float]:
    return [round(value, 4) for value in point]


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _rotate_z(point: tuple[float, float, float], degrees: float) -> tuple[float, float, float]:
    radians = math.radians(degrees)
    x, y, z = point
    return (
        x * math.cos(radians) - y * math.sin(radians),
        x * math.sin(radians) + y * math.cos(radians),
        z,
    )


def _translate(
    point: tuple[float, float, float],
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(value + delta for value, delta in zip(point, offset))


def _transform(
    point: tuple[float, float, float],
    rotation_deg: float,
    translation: tuple[float, float, float],
) -> tuple[float, float, float]:
    return _translate(_rotate_z(point, rotation_deg), translation)


def _inverse_transform(
    point: tuple[float, float, float],
    rotation_deg: float,
    translation: tuple[float, float, float],
) -> tuple[float, float, float]:
    shifted = tuple(value - delta for value, delta in zip(point, translation))
    return _rotate_z(shifted, -rotation_deg)


def _residual_noise(index: int, fragment_index: int) -> tuple[float, float, float]:
    # Small deterministic residual that imitates post-ICP measurement error.
    scale = 0.006 + fragment_index * 0.0015
    return (
        math.sin(index * 1.7 + fragment_index) * scale,
        math.cos(index * 1.3 + fragment_index * 0.5) * scale,
        math.sin(index * 0.9) * scale * 0.35,
    )


def _fragment_points(
    x_min: float,
    x_max: float,
    fragment_index: int,
    rows: int = 12,
    cols: int = 7,
) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for row in range(rows):
        y = -1.25 + row * (2.5 / (rows - 1))
        for col in range(cols):
            x = x_min + col * ((x_max - x_min) / (cols - 1))
            edge_wave = 0.025 * math.sin(row * 1.9 + fragment_index)
            if col == 0:
                x += edge_wave
            if col == cols - 1:
                x -= edge_wave * 0.8
            relief = 0.045 * math.sin(x * 4.1) * math.cos(y * 2.7)
            points.append((x, y, relief))
    return points


def build_fixture() -> dict:
    fragment_specs = [
        {
            "id": "left_shard",
            "label": "Left fracture shard",
            "color": "#2563eb",
            "x_range": (-1.12, -0.32),
            "rotation_deg": -32.0,
            "translation": (-0.92, 0.52, 0.0),
        },
        {
            "id": "center_shard",
            "label": "Center relief shard",
            "color": "#d97706",
            "x_range": (-0.38, 0.34),
            "rotation_deg": 18.0,
            "translation": (0.24, -0.44, 0.0),
        },
        {
            "id": "right_shard",
            "label": "Right edge shard",
            "color": "#059669",
            "x_range": (0.28, 1.08),
            "rotation_deg": 41.0,
            "translation": (0.88, 0.38, 0.0),
        },
    ]

    fragments = []
    before_errors = []
    after_errors = []

    for fragment_index, spec in enumerate(fragment_specs):
        target_points = _fragment_points(*spec["x_range"], fragment_index)
        scrambled_points = [
            _transform(point, spec["rotation_deg"], spec["translation"])
            for point in target_points
        ]

        aligned_points = []
        for point_index, point in enumerate(scrambled_points):
            aligned = _inverse_transform(point, spec["rotation_deg"], spec["translation"])
            aligned = _translate(aligned, _residual_noise(point_index, fragment_index))
            aligned_points.append(aligned)

        fragment_before = [
            _distance(scrambled, target)
            for scrambled, target in zip(scrambled_points, target_points)
        ]
        fragment_after = [
            _distance(aligned, target)
            for aligned, target in zip(aligned_points, target_points)
        ]
        before_errors.extend(fragment_before)
        after_errors.extend(fragment_after)

        fragments.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "color": spec["color"],
                "target": [_round_point(point) for point in target_points],
                "scrambled": [_round_point(point) for point in scrambled_points],
                "aligned": [_round_point(point) for point in aligned_points],
                "estimated_transform": {
                    "rotation_z_degrees": -spec["rotation_deg"],
                    "translation": _round_point(tuple(-value for value in spec["translation"])),
                },
                "metrics": {
                    "rms_before": round(mean(fragment_before), 4),
                    "rms_after": round(mean(fragment_after), 4),
                    "point_count": len(target_points),
                },
            }
        )

    rms_before = mean(before_errors)
    rms_after = mean(after_errors)
    return {
        "metadata": {
            "name": "Healing Stones synthetic alignment fixture",
            "description": "Small deterministic artifact shards for the browser demo.",
            "fragment_count": len(fragments),
            "point_count": sum(len(fragment["target"]) for fragment in fragments),
            "units": "normalized artifact coordinates",
        },
        "metrics": {
            "mean_residual_before": round(rms_before, 4),
            "mean_residual_after": round(rms_after, 4),
            "improvement_ratio": round(rms_before / rms_after, 2),
        },
        "fragments": fragments,
    }


def write_fixture(output_path: Path, report_path: Path) -> dict:
    fixture = build_fixture()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")

    report = {
        "summary": fixture["metadata"],
        "metrics": fixture["metrics"],
        "fragments": [
            {
                "id": fragment["id"],
                "label": fragment["label"],
                "metrics": fragment["metrics"],
                "estimated_transform": fragment["estimated_transform"],
            }
            for fragment in fixture["fragments"]
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return fixture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixture = write_fixture(args.output, args.report)
    print(f"Wrote browser fixture: {args.output}")
    print(f"Wrote metrics report:  {args.report}")
    print(
        "Residual improvement: "
        f"{fixture['metrics']['mean_residual_before']} -> "
        f"{fixture['metrics']['mean_residual_after']} "
        f"({fixture['metrics']['improvement_ratio']}x)"
    )


if __name__ == "__main__":
    main()
