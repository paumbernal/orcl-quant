"""Command-line entry point:  ``orcl-lab build | analyse | report | all``."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="orcl-lab", description="ORCL Quantitative Investment & AI Valuation Lab")
    ap.add_argument("command", choices=["build", "analyse", "report", "charts", "all"],
                    help="build = download data; analyse = run every model; report = write the research report; "
                         "charts = export static PNG charts; all = everything")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--cache", action="store_true", help="re-use data/raw instead of re-downloading")
    ap.add_argument("--steps", nargs="*", default=None, help="subset of build steps")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.command in ("build", "all"):
        from .data.pipeline import STEPS, run_pipeline
        run_pipeline(cfg, args.steps or STEPS, use_cache=args.cache)
        print("data pipeline complete -> data/processed, data/manifest.json")
    if args.command in ("analyse", "all"):
        from .reporting.runner import run_all_analyses
        run_all_analyses(cfg)
        print("analyses complete -> models/")
    if args.command in ("report", "all"):
        from .reporting.report import write_report
        path = write_report(cfg)
        print(f"report written -> {path}")
    if args.command in ("charts", "all"):
        from .viz.export import export_all
        n = export_all(cfg)
        print(f"{n} charts exported -> visualisations/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
