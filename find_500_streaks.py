#!/usr/bin/env python

import mlb_standings

SPECIAL_TEAM_NAMES = {
    "Oakland Athletics": "Athletics",
    "Cleveland Indians": "Cleveland Guardians",
    "Florida Marlins": "Miami Marlins",
    "Tampa Bay Devil Rays": "Tampa Bay Rays",
    "Anaheim Angels": "Los Angeles Angels",
    "California Angels": "Los Angeles Angels",
    "Montreal Expos": "Washington Nationals",
}

FRANCHISE_STARTS = {"Colorado Rockies": 1993}


def find_500_streaks(years, data_path="data"):
    streak_length = {}
    over_500 = {}
    finished = {}
    for year in years:
        print(f"Beginning year {year}...")
        played, _ = mlb_standings.statsapi_schedule_to_played_unplayed(
            f"{data_path}/schedule_{year}_1.json"
        )
        played.sort_values(by="completion_date", ascending=False, inplace=True)
        for _, row in played.iterrows():
            home = SPECIAL_TEAM_NAMES.get(row["home"], row["home"])
            visitor = SPECIAL_TEAM_NAMES.get(row["visitor"], row["visitor"])
            # print(f"Considering the {row['completion_date']} {visitor}@{home} game...")
            if row["outcome"] == mlb_standings.TIE:
                for team in [home, visitor]:
                    if not finished.get(team):
                        print(
                            f"The {team} had a tie on {row['completion_date']} so their streak might be an odd number."
                        )
                        streak_length[team] = streak_length[team] + 1
                continue
            if row["outcome"] == mlb_standings.HOME_WON:
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
                    if len(finished) >= 30:
                        return
        for team in streak_length:
            if not finished.get(team):
                print(
                    f"After processing {year} the {team} are still {over_500[team]} games over .500."
                )
                if FRANCHISE_STARTS.get(team) == year:
                    print(
                        f"The {team} have no .500 streak going back to the beginning of their franchise."
                    )
                    finished[team] = True
    for team in streak_length:
        if not finished.get(team):
            print(f"The {team}' shortest .500 streak began before {years[-1]}.")


if __name__ == "__main__":
    find_500_streaks(range(2025, 1981, -1), data_path="./non_git_data")
