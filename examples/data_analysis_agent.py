"""A data analysis agent that computes its answers in a sandbox instead of guessing.

Models are unreliable at arithmetic over raw data pasted into context, so this
agent never sees the dataset: the data directory is mounted read-only into the
`CodeMode` sandbox, where the model computes its aggregates with real Python.

The demo generates a synthetic orders CSV and asks for an analysis:

    uv run examples/data_analysis_agent.py

Point it at your own data instead:

    uv run examples/data_analysis_agent.py path/to/data-directory "your question"

Requires the `code-mode` extra.
"""

import csv
import os
import random
import sys
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_monty import MountDir

from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.overflowing_tool_output import OverflowingToolOutput

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-opus-4-7')

INSTRUCTIONS = """\
You are a data analyst. The dataset directory is mounted read-only at `/data`
inside your sandbox: load files with `pathlib`. The sandbox is a restricted
Python without the `csv` module, so parse lines manually (all parsed fields
are strings).

- Never estimate a number you can compute. Load the data and compute it in `run_code`.
- Verify your parsing before aggregating: print one parsed row next to its raw line
  and check every field survived intact (watch delimiters and line endings).
- Show your working: report which file, how many rows, and the exact aggregation used.
- Sanity-check results (totals reconcile, percentages sum to ~100) before reporting.
- If the data can't answer the question, say what's missing instead of approximating.
"""


def build_agent(model: Model | str = DEFAULT_MODEL, data_dir: Path | None = None) -> Agent:
    """Build the analysis agent scoped to `data_dir`."""
    data_dir = data_dir or Path.cwd()
    return Agent(
        model,
        capabilities=[
            # The sandbox sees the data directory at /data (read-only) and nothing
            # else on the machine. The model loads and aggregates with real Python,
            # so the numbers are computed, not guessed -- and the raw rows never
            # enter the context window.
            CodeMode(mount=MountDir(virtual_path='/data', host_path=str(data_dir), mode='read-only')),
            OverflowingToolOutput(),
        ],
        instructions=INSTRUCTIONS,
    )


def write_demo_orders(directory: Path, rows: int = 500) -> Path:
    """Write a deterministic synthetic orders CSV to analyze."""
    rng = random.Random(0)
    path = directory / 'orders.csv'
    regions = ['north', 'south', 'east', 'west']
    with path.open('w', newline='') as f:
        # LF terminators: csv.writer's default \r\n is a portability trap for
        # consumers that split on \n (the trailing \r corrupts the last field).
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['order_id', 'region', 'units', 'unit_price', 'returned'])
        for i in range(rows):
            writer.writerow(
                [
                    i + 1,
                    rng.choice(regions),
                    rng.randint(1, 20),
                    round(rng.uniform(5, 200), 2),
                    int(rng.random() < 0.07),
                ]
            )
    return path


def main() -> None:
    """Analyze the given data directory, or a generated demo dataset."""
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
        question = ' '.join(sys.argv[2:]) or 'Summarize this dataset: shape, notable distributions, and anomalies.'
    else:
        data_dir = Path(tempfile.mkdtemp(prefix='analysis-demo-'))
        write_demo_orders(data_dir)
        print(f'demo dataset written to {data_dir}')
        question = (
            'Using orders.csv: total revenue by region (units * unit_price, excluding returns), '
            'the return rate per region, and whether any region is an outlier.'
        )
    print(build_agent(data_dir=data_dir).run_sync(question).output)


if __name__ == '__main__':
    main()
