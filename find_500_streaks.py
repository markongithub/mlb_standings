#!/usr/bin/env python

import mlb_standings
import pandas as pd

FRANCHISE_STARTS = {"COL": 1993}


def find_500_streaks(years, data_path="data"):
    NICKNAMES = mlb_standings.load_nicknames(f"{data_path}/CurrentNames.csv")
    # CurrentNames.csv does not, as of June 1, 2025, know that the Athletics are
    # currently experiencing homelessness.
    statsapi_name_to_franchise = {"Athletics": "OAK"}
    retrosheet_name_to_franchise = {}
    NICKNAMES["start_date"] = pd.to_datetime(NICKNAMES["start_date"])
    franchise_to_latest_name = {}
    for _, row in NICKNAMES.sort_values("start_date").iterrows():
        statsapi_name = f"{row['location']} {row['nickname']}"
        print(f"Adding the {statsapi_name} to my team dictionary...")
        statsapi_name_to_franchise[statsapi_name] = row["current_id"]
        retrosheet_name_to_franchise[row["contemporary_id"]] = row["current_id"]
        franchise_to_latest_name[row["current_id"]] = statsapi_name
    franchise_to_latest_name["OAK"] = "Athletics"
    streak_length = {}
    over_500 = {}
    finished = {}
    for year in years:
        print(f"Beginning year {year}...")
        from_statsapi = False
        if year > 2021:
            played, _ = mlb_standings.statsapi_schedule_to_played_unplayed(
                f"{data_path}/schedule_{year}_1.json"
            )
            from_statsapi = True
        else:
            game_log, schedule = mlb_standings.retrosheet_game_log_and_schedule(
                year, data_path
            )
            played, _ = mlb_standings.retrosheet_to_played_unplayed(
                game_log, schedule, year
            )
        played.sort_values(by="completion_date", ascending=False, inplace=True)
        for _, row in played.iterrows():
            if from_statsapi:
                home = statsapi_name_to_franchise[row["home"]]
                visitor = statsapi_name_to_franchise[row["visitor"]]
            else:
                home = retrosheet_name_to_franchise[row["home"]]
                visitor = retrosheet_name_to_franchise[row["visitor"]]
            # print(f"Considering the {row['completion_date']} {visitor}@{home} game...")
            if row["outcome"] == mlb_standings.TIE:
                for team in [home, visitor]:
                    if not finished.get(team):
                        print(
                            f"The {franchise_to_latest_name[team]} had a tie on {row['completion_date']} so their streak might be an odd number."
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
                        f"The {franchise_to_latest_name[team]}' .500 streak of {streak_length[team]} games began on {row['completion_date']}"
                    )
                    finished[team] = True
                    if len(finished) >= 30:
                        return
        for team in streak_length:
            if not finished.get(team):
                print(
                    f"After processing {year} the {franchise_to_latest_name[team]} are still {over_500[team]} games over .500."
                )
                if FRANCHISE_STARTS.get(team) == year:
                    print(
                        f"The {franchise_to_latest_name[team]} have no .500 streak going back to the beginning of their franchise."
                    )
                    finished[team] = True
    for team in streak_length:
        if not finished.get(team):
            print(
                f"The {franchise_to_latest_name[team]}' shortest .500 streak began before {years[-1]}."
            )


if __name__ == "__main__":
    find_500_streaks(range(2025, 1903, -1), data_path="./non_git_data")
