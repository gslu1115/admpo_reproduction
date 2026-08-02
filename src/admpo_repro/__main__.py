from __future__ import annotations

import argparse
import json

from admpo_repro.runtime import configure_mujoco_environment


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="admpo-repro")
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="运行 Figure 2 或 Figure 4")
    run.add_argument("experiment", choices=("figure2", "figure4"))
    run.add_argument(
        "--phase", choices=("smoke", "pilot", "deadline48h", "full"), default="smoke"
    )
    run.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    run.add_argument("--resume", action="store_true")
    plot = commands.add_parser("plot", help="从 CSV 重绘论文风格图")
    plot.add_argument("experiment", choices=("figure2", "figure4"))
    plot.add_argument("--phase", choices=("deadline48h", "full"), default="full")
    check = commands.add_parser("check", help="检查环境、数据集和 MuJoCo oracle")
    check.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    return root


def main() -> None:
    args = parser().parse_args()
    configure_mujoco_environment()
    # Import after setting LD_LIBRARY_PATH so mujoco-py sees the current process configuration.
    from admpo_repro.runner import check_environment, plot_experiment, run_experiment

    if args.command == "run":
        run_experiment(args.experiment, args.phase, args.seeds, args.resume)
    elif args.command == "plot":
        png, pdf = plot_experiment(args.experiment, args.phase)
        print(json.dumps({"png": str(png), "pdf": str(pdf)}, ensure_ascii=False))
    else:
        print(json.dumps(check_environment(args.phase), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
