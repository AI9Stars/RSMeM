import argparse
import asyncio
import sys

from rsmem.config import ExperimentConfig
from rsmem.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rsmem",
        description="RSMeM",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument("--config", required=True, help="Path to YAML config file")
    run_parser.add_argument(
        "--resume-from",
        default=None,
        dest="resume_from",
        help="Resume from previous attempt directory",
    )
    run_parser.add_argument(
        "--continue-attempt",
        default=None,
        dest="continue_attempt",
        help="Continue a partial attempt in-place (re-run only error questions)",
    )
    run_parser.add_argument(
        "--resume-through-attempt",
        type=int,
        default=1,
        dest="resume_through_attempt",
        help="Number of completed attempts to load from resume source (default: 1)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "run":
        config = ExperimentConfig.from_yaml(args.config)
        asyncio.run(run_experiment(
            config,
            resume_from_dir=args.resume_from,
            continue_from_dir=args.continue_attempt,
            resume_through_attempt=args.resume_through_attempt,
        ))
    else:
        build_parser().print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
