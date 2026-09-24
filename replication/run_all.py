"""Master script: reproduces every table and figure in the paper from the raw data.

Usage
-----
    python run_all.py              full run (reproduces the published numbers)
    python run_all.py --fast       quick run (fewer randomisation draws)
    python run_all.py --stage 2    run a single stage (1=data, 2=tables,
                                   3=DML, 4=inference, 5=figures)

Outputs land in ./results/ as both plain-text tables and machine-readable CSVs.
"""
import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as C  # noqa: E402


BANNER = r"""
--------------------------------------------------------------------------
  Growing-season climate variability and food prices
  Replication package - runs end to end from the two raw CSV files
--------------------------------------------------------------------------
"""


def check_environment():
    print("Environment")
    print("  python      : {}.{}.{}".format(*sys.version_info[:3]))
    ok = True
    for mod, minver in [("numpy", "1.24"), ("pandas", "2.0"),
                        ("scipy", "1.10"), ("sklearn", "1.2"),
                        ("statsmodels", "0.14")]:
        try:
            m = __import__(mod)
            print("  {:<12}: {}".format(mod, getattr(m, "__version__", "?")))
        except ImportError:
            print("  {:<12}: MISSING  (pip install -r requirements.txt)".format(mod))
            ok = False
    if not ok:
        raise SystemExit("\nInstall the missing packages and re-run.")
    C.check_inputs()
    print("  data folder : {}".format(C.RAW_DIR))
    print("  results     : {}".format(C.RESULTS))
    print("  mode        : {}".format("FAST (fewer RI draws)" if C.FAST else "FULL"))
    print("")


def run_stage(n):
    t0 = time.time()
    if n == 1:
        print(">>> STAGE 1/5  building the analysis panel")
        import rep_data
        rep_data.build_panel()
    elif n == 2:
        print(">>> STAGE 2/5  Tables 1-6: descriptives, IRFs, convexity, mechanism,")
        print("               heterogeneity, robustness")
        import rep_tables
        rep_tables.main()
    elif n == 3:
        print(">>> STAGE 3/5  Table 7: double machine learning")
        import rep_dml
        rep_dml.main()
    elif n == 4:
        print(">>> STAGE 4/5  Tables 8-14: balance, diagnostics, stability,")
        print("               jackknife, randomisation inference, volatility trend")
        import rep_inference
        rep_inference.main()
    elif n == 5:
        print(">>> STAGE 5/5  Figures 1-4 (PNG and PDF)")
        import rep_figures
        rep_figures.main()
    print("    stage {} finished in {:.1f}s\n".format(n, time.time() - t0))


def main():
    ap = argparse.ArgumentParser(description="Replication package driver")
    ap.add_argument("--fast", action="store_true",
                    help="fewer randomisation-inference draws (quick check)")
    ap.add_argument("--stage", type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help="run only this stage")
    args = ap.parse_args()

    if args.fast:
        os.environ["REPFAST"] = "1"
        C.FAST = True
        C.N_PERM = 200
        C.RI_TIME_BUDGET = 120

    C.ascii_stdout()
    C.ensure_dirs()
    print(BANNER)
    check_environment()

    stages = [args.stage] if args.stage else [1, 2, 3, 4, 5]
    t0 = time.time()
    for n in stages:
        try:
            run_stage(n)
        except Exception:
            print("\n!!! stage {} failed:\n".format(n))
            traceback.print_exc()
            raise SystemExit(1)

    print("-" * 74)
    print("  All stages completed in {:.1f} minutes.".format((time.time() - t0) / 60))
    print("  Text tables : results/tables.txt, results/dml.txt, results/diagnostics.txt")
    print("  CSV tables  : results/table*.csv")
    print("-" * 74)


if __name__ == "__main__":
    main()
