"""CLI wrapper for the Llama WikiText-2 PPL experiment."""

from __future__ import annotations

import sys
from pathlib import Path

# Make direct source-checkout execution work without an editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nestquant.experiments.llm_perplexity import main


if __name__ == "__main__":
    main()
