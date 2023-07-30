#!/usr/bin/env python

import mlb_standings
import sys

if __name__ == "__main__":
    try:
        data_path = sys.argv[2]
    except IndexError:
        data_path='./data'
    mlb_standings.run_one_year_retro(int(sys.argv[1]), data_path)
