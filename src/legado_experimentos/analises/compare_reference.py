"""Compara um PL4 do circuito validado com um ADF exportado pelo PlotXY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from signal_io import compare_signals, read_canonical_pl4, read_reference_adf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pl4", type=Path)
    parser.add_argument("adf", type=Path)
    args = parser.parse_args()

    pl4 = read_canonical_pl4(args.pl4)
    adf = read_reference_adf(args.adf)
    result = compare_signals(pl4, adf)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
