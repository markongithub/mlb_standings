#!/usr/bin/env python

import mlb_standings
import sys

if __name__ == "__main__":
    mlb_standings.run_one_year_retro(int(sys.argv[1]))
