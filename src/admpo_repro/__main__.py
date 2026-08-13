from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="admpo-repro")
    commands = root.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run a reproduction experiment")
    experiments = run.add_subparsers(dest="experiment", required=True)

    figure2 = experiments.add_parser("figure2", help="Run the Figure 2 pipeline")
    figure2.add_argument(
        "--phase",
        choices=("smoke", "pilot", "deadline48h", "full"),
        default="smoke",
    )
    figure2.add_argument("--seeds", type=int, nargs="+", default=None)
    figure2.add_argument("--workers", type=int, default=0)
    figure2.add_argument("--resume", action="store_true")
    figure2.add_argument("--namespace", default=None)

    experiments.add_parser(
        "figure4", help="Run the fixed Figure 4 main evaluation (Hopper, h=5, 5000 windows)"
    )

    plot = commands.add_parser("plot", help="Replot Figure 2 from saved CSV files")
    plot.add_argument("experiment", choices=("figure2",))
    plot.add_argument(
        "--phase", choices=("smoke", "pilot", "deadline48h", "full"), default="full"
    )
    plot.add_argument("--namespace", default=None)

    check = commands.add_parser(
        "check", help="Check the cached datasets and Figure 2 trajectory split"
    )
    check.add_argument("--phase", choices=("smoke", "full"), default="smoke")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "run" and args.experiment == "figure4":
        from admpo_repro.evaluation.figure4_uncertainty.run_evaluation import run_phase

        output_dir, _ = run_phase(
            reexec_module="admpo_repro",
            reexec_args=sys.argv[1:],
        )
        print(
            json.dumps(
                {"status": "complete", "output": str(output_dir)},
                ensure_ascii=False,
            )
        )
        return

    from admpo_repro.runner import check_environment, plot_experiment, run_experiment

    if args.command == "run":
        run_experiment(
            args.experiment,
            args.phase,
            args.seeds,
            args.resume,
            args.workers,
            args.namespace,
        )
    elif args.command == "plot":
        png, pdf = plot_experiment(args.experiment, args.phase, args.namespace)
        print(json.dumps({"png": str(png), "pdf": str(pdf)}, ensure_ascii=False))
    else:
        print(json.dumps(check_environment(args.phase), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
