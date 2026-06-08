#!/usr/bin/env python3
"""Generate DLIO run-state plots from run_stats.csv.

The script intentionally depends only on Python's standard library plus
matplotlib so it can run inside the existing ROS Docker image.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PLOTS = [
    (
        "pose_position",
        "Robot position in dlio_map",
        "position [m]",
        [
            ("p_x_m", "x"),
            ("p_y_m", "y"),
            ("p_z_m", "z"),
        ],
    ),
    (
        "pose_orientation_rpy",
        "Robot orientation in dlio_map",
        "angle [rad]",
        [
            ("roll_rad", "roll"),
            ("pitch_rad", "pitch"),
            ("yaw_rad", "yaw"),
        ],
    ),
    (
        "twist_linear_body",
        "Body-frame linear velocity",
        "linear velocity [m/s]",
        [
            ("vlin_b_x_mps", "vx"),
            ("vlin_b_y_mps", "vy"),
            ("vlin_b_z_mps", "vz"),
        ],
    ),
    (
        "twist_angular_body",
        "Body-frame angular velocity",
        "angular velocity [rad/s]",
        [
            ("vang_b_x_radps", "wx"),
            ("vang_b_y_radps", "wy"),
            ("vang_b_z_radps", "wz"),
        ],
    ),
    (
        "bias_accel",
        "Estimated accelerometer bias",
        "accel bias [m/s^2]",
        [
            ("accel_bias_x_mps2", "bx"),
            ("accel_bias_y_mps2", "by"),
            ("accel_bias_z_mps2", "bz"),
        ],
    ),
    (
        "bias_gyro",
        "Estimated gyroscope bias",
        "gyro bias [rad/s]",
        [
            ("gyro_bias_x_radps", "bx"),
            ("gyro_bias_y_radps", "by"),
            ("gyro_bias_z_radps", "bz"),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Input run_stats.csv")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--dpi", default=600, type=int, help="PNG output DPI")
    return parser.parse_args()


def read_csv(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"{csv_path} does not have a CSV header")

        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                columns[name].append(float(value))

    if not columns.get("time_s"):
        raise RuntimeError(f"{csv_path} has no data rows")
    return columns


def require_columns(columns: dict[str, list[float]], names: Iterable[str]) -> None:
    missing = [name for name in names if name not in columns]
    if missing:
        raise RuntimeError(f"Missing required column(s): {', '.join(missing)}")


def write_plot(
    columns: dict[str, list[float]],
    out_dir: Path,
    dpi: int,
    basename: str,
    title: str,
    ylabel: str,
    series: list[tuple[str, str]],
) -> None:
    require_columns(columns, ["time_s", *[name for name, _ in series]])
    time_s = columns["time_s"]

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for column, label in series:
        ax.plot(time_s, columns[column], linewidth=1.4, label=label)

    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", frameon=True)

    fig.savefig(out_dir / f"{basename}.pdf")
    fig.savefig(out_dir / f"{basename}.png", dpi=dpi)
    plt.close(fig)


def write_summary(columns: dict[str, list[float]], out_dir: Path) -> None:
    time_s = columns["time_s"]
    with (out_dir / "run_stats_summary.txt").open("w") as handle:
        handle.write(f"rows={len(time_s)}\n")
        handle.write(f"time_start_s={time_s[0]:.9f}\n")
        handle.write(f"time_end_s={time_s[-1]:.9f}\n")
        handle.write(f"duration_s={time_s[-1] - time_s[0]:.9f}\n")


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = read_csv(args.csv)

    for basename, title, ylabel, series in PLOTS:
        write_plot(columns, args.out_dir, args.dpi, basename, title, ylabel, series)

    write_summary(columns, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
