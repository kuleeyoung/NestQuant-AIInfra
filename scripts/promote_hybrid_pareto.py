"""CLI wrapper for promoting hybrid-residency Pareto evidence."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nestquant.cli.promote_hybrid_pareto import main


if __name__ == "__main__":
    main()
