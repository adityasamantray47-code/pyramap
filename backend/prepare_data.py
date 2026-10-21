"""
Downloads real NASA/NOAA data and caches it to data_cache/states/<STATE>.csv.gz.

Usage:
  python prepare_data.py --states CO
  python prepare_data.py --states CO,NM,AZ
  python prepare_data.py              # all 50 states
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv

from grid import ALL_STATES
from real_data import assemble

load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=str, default=None, help="Comma-separated state codes, e.g. CO,NM. Default: all 50.")
    args = parser.parse_args()

    map_key = os.environ.get("FIRMS_MAP_KEY")
    if not map_key:
        print("ERROR: FIRMS_MAP_KEY not set. Copy backend/.env.example to backend/.env and fill it in.")
        sys.exit(1)

    states = [s.strip().upper() for s in args.states.split(",")] if args.states else ALL_STATES

    for i, state in enumerate(states, start=1):
        print(f"[{i}/{len(states)}] {state}: downloading...")
        t0 = time.time()
        try:
            df = assemble.assemble_state(state, map_key)
            if df.empty:
                print(f"  {state}: no data assembled, skipping save.")
                continue
            path = assemble.save_state(state, df)
            print(f"  {state}: saved {len(df)} rows to {path} ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"  {state}: FAILED - {exc}")


if __name__ == "__main__":
    main()
