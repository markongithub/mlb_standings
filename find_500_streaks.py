#!/usr/bin/env python

import mlb_standings

HOME_WON = 0
VISITOR_WON = 1
TIE = 2

SPECIAL_TEAM_NAMES = {"Oakland Athletics": "Athletics"}


def find_500_streaks(year, data_path="data"):
    played, _ = mlb_standings.statsapi_schedule_to_played_unplayed(
        f"{data_path}/schedule_{year}_1.json"
    )
    streak_length = {}
    over_500 = {}
    finished = {}
    played.sort_values(by="completion_date", ascending=False, inplace=True)
    for _, row in played.iterrows():
        home = SPECIAL_TEAM_NAMES.get(row["home"], row["home"])
        visitor = SPECIAL_TEAM_NAMES.get(row["visitor"], row["visitor"])
        # print(f"Considering the {row['completion_date']} {visitor}@{home} game...")
        if row["outcome"] == TIE:
            print("A tie? That's stupid.")
            continue
        if row["outcome"] == HOME_WON:
            winner = home
            loser = visitor
        else:
            winner = visitor
            loser = home
        if not finished.get(winner):
            over_500[winner] = over_500.get(winner, 0) + 1
            streak_length[winner] = streak_length.get(winner, 0) + 1
        if not finished.get(loser):
            over_500[loser] = over_500.get(loser, 0) - 1
            streak_length[loser] = streak_length.get(loser, 0) + 1
        for team in [winner, loser]:
            if not finished.get(team) and over_500[team] == 0:
                print(
                    f"The {team}' .500 streak of {streak_length[team]} games began on {row['completion_date']}"
                )
                finished[team] = True
    for team in streak_length:
        if not finished.get(team):
            print(f"The {team}' shortest .500 streak began before this season.")


if __name__ == "__main__":
    find_500_streaks(2025, data_path="./non_git_data")
