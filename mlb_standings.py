#!/usr/bin/env python

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class SeasonParameters(object):
    def __init__(
        self,
        year,
        nicknames=None,
        team_ids=None,
        retrosheet_schedule=None,
        statsapi_played=None,
        statsapi_unplayed=None,
        statsapi_teams=None,
    ):
        if not (
            (
                statsapi_teams is not None
                and statsapi_played is not None
                and statsapi_unplayed is not None
            )
            or (nicknames is not None and team_ids is not None)
        ):
            raise ("I feel like you passed in the wrong input.")

        self.year = year
        self.wildcard_count = wildcards_for_year(year)
        self.winners_per_division = get_winners_per_division(year)
        if statsapi_teams:
            self.divisions = divisions_from_statsapi_teams(statsapi_teams)
        else:
            self.divisions = divisions_for_year(nicknames, team_ids, year)
        if retrosheet_schedule is not None:
            self.season_lengths = get_season_lengths(retrosheet_schedule)
        elif statsapi_played is not None and statsapi_unplayed is not None:
            self.season_lengths = get_season_lengths_statsapi(
                statsapi_played, statsapi_unplayed, self.divisions
            )
        else:
            self.season_lengths = pd.Series(data=[162, 162], index=["NL", "AL"])
            # [['NL', 162], ['AL', 162]], columns=['lg', 'length'])
        self.tiebreakers_required = year >= 2022

    def use_half_seasons(self, team):
        if self.year == 1981:
            return True
        league = self.divisions.loc[team]["lg"]
        return league in [
            "Eastern League",
            "Southern League",
            "Texas League",
            "Midwest League",
            "Northwest League",
            "South Atlantic League",
            "California League",
            "Florida State League",
        ]


def load_game_log(game_log_path):
    retro_df_columns = [
        "date",
        "doubleheader_index",
        "weekday",
        "visitor",
        "visitor_league",
        "visitor_game_num",
        "home",
        "home_league",
        "home_game_num",
        "visitor_score",
        "home_score",
        "length_outs",
        "day_night",
        "completion",
        "forfeit",
    ]
    return pd.read_csv(
        game_log_path,
        header=None,
        names=retro_df_columns,
        usecols=range(len(retro_df_columns)),
    )


#    df['date'] = df['date'].astype(str)
#    df = df.fillna(value={'completion': 'I hate Pandas'})
#    df['completion'] = df['completion'].astype(str)
#    df['completion_date'] = df['date']
#    df.loc[(df['completion'] != "I hate Pandas"), 'completion_date'] = df['completion'].str.split(',').str[0]
#    df['completion_date'] = df['completion_date'].astype(str)
#    return df


def fix_1880(schedule):
    # Baseball Reference calls them the Reds so I am going to coerce this code
    # to call them the Reds.
    schedule.loc[schedule["visitor"] == "CN1", "visitor"] = "CN4"
    schedule.loc[schedule["home"] == "CN1", "home"] = "CN4"
    return schedule


def fix_1882(game_log):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log["visitor"] == "BL5", "visitor"] = "BL2"
    game_log.loc[game_log["home"] == "BL5", "home"] = "BL2"
    return game_log


def fix_1884(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log = game_log.loc[
        (game_log["visitor_league"] != "UA") & (game_log["home_league"] != "UA")
    ]
    game_log.loc[game_log["visitor"] == "WS7", "visitor"] = "RIC"
    game_log.loc[game_log["home"] == "WS7", "home"] = "RIC"
    schedule = schedule.loc[
        (schedule["visitor_league"] != "UA") & (schedule["home_league"] != "UA")
    ]
    schedule.loc[schedule["visitor"] == "WS7", "visitor"] = "RIC"
    schedule.loc[schedule["home"] == "WS7", "home"] = "RIC"
    return game_log, schedule


def fix_1890(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log["visitor"] == "BR4", "visitor"] = "BL3"
    game_log.loc[game_log["home"] == "BR4", "home"] = "BL3"
    schedule.loc[schedule["visitor"] == "BR4", "visitor"] = "BL3"
    schedule.loc[schedule["home"] == "BR4", "home"] = "BL3"
    return game_log, schedule


def fix_1891(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log["visitor"] == "CN3", "visitor"] = "ML3"
    game_log.loc[game_log["home"] == "CN3", "home"] = "ML3"
    schedule.loc[schedule["visitor"] == "CN3", "visitor"] = "ML3"
    schedule.loc[schedule["home"] == "CN3", "home"] = "ML3"
    return game_log, schedule


# adapted from sdvinay except now totally different
def compute_standings(game_log):
    gms_played = game_log.copy()
    # We'll have to handle ties and forfeits elsewhere for Retrosheet.
    winners = pd.Series(
        np.where(gms_played["home_won"], gms_played["home"], gms_played["visitor"])
    )
    losers = pd.Series(
        np.where(gms_played["home_won"], gms_played["visitor"], gms_played["home"])
    )
    standings = pd.concat(
        [winners.value_counts().rename("W"), losers.value_counts().rename("L")], axis=1
    )
    return standings.fillna(0)


# get this from https://www.retrosheet.org/Nickname.htm
# It doesn't have a lot of early stuff.
def load_nicknames(nicknames_file):
    retro_nickname_columns = [
        "current_id",
        "contemporary_id",
        "league",
        "division",
        "location",
        "nickname",
        "alt_nickname",
        "start_date",
        "end_date",
    ]
    df = pd.read_csv(
        nicknames_file,
        header=None,
        names=retro_nickname_columns,
        usecols=range(len(retro_nickname_columns)),
    )
    return df


def load_team_ids(team_ids_file):
    retro_team_id_columns = [
        "team_id",
        "league",
        "location",
        "nickname",
        "start_year",
        "end_year",
    ]
    df = pd.read_csv(
        team_ids_file,
        header=None,
        names=retro_team_id_columns,
        usecols=range(len(retro_team_id_columns)),
    )
    return df


def nicknames_to_team_ids(divisions):
    output = {}
    for index, row in divisions.iterrows():
        if row["nickname"] == "Indians":
            full_nickname = f'{row["location"]} Guardians'
        else:
            full_nickname = f'{row["location"]} {row["nickname"]}'
        output[full_nickname] = index
    return output


def divisions_for_year(nicknames_from_retrosheet, team_ids_from_retrosheet, year):
    if year <= 1915:
        return divisions_from_team_ids(team_ids_from_retrosheet, year)
    df = nicknames_from_retrosheet.copy()
    # df['start_date'] = df['start_date'].astype(str)
    df["start_year"] = pd.DatetimeIndex(df["start_date"]).year
    df["end_year"] = pd.DatetimeIndex(df["end_date"]).year
    df = df.loc[
        (df["start_year"] <= year)
        & ((df["end_year"].isna()) | (df["end_year"] >= year))
    ]  # & (df['end_year'] => year)]
    df["div"] = df[["league", "division"]].fillna("").sum(axis=1)

    return df.set_index("contemporary_id")[
        ["league", "div", "current_id", "location", "nickname"]
    ].rename(columns={"league": "lg", "current_id": "franchise_id"})


def divisions_from_team_ids(team_ids_from_retrosheet, year):
    df = team_ids_from_retrosheet.copy()
    df = df.loc[(df["start_year"] <= year) & (df["end_year"] >= year)]
    df["div"] = df["league"]  # the league is the division is the league
    df["franchise_id"] = df["team_id"]
    return df.set_index("team_id")[
        ["league", "div", "franchise_id", "location", "nickname"]
    ].rename(columns={"league": "lg"})


def divisions_from_statsapi_teams(statsapi_teams):
    tuples = [
        (
            team["name"],
            team["league"]["name"],
            team["division"]["name"],
            team["locationName"],
            team["teamName"],
            "whatevz",
        )
        for team in statsapi_teams["teams"]
        if team.get("division")
    ]
    return pd.DataFrame.from_records(
        np.array(
            tuples,
            dtype=[
                ("name", "O"),
                ("lg", "O"),
                ("div", "O"),
                ("location", "O"),
                ("nickname", "O"),
                ("franchise_id", "O"),
            ],
        )
    ).set_index("name")


def division_contenders(standings_immutable, season_length, division_winners=1):
    df = standings_immutable.copy()
    #    print(f'division_contenders was called with columns {df.columns}')
    # df['div'] = divisions['div']
    df["max_wins"] = df["season_length"] - df["L"]
    division_contenders = set()
    division_win_threshold = {}
    for division in df["div"].unique():
        division_win_threshold[division] = (
            df.loc[(df["div"] == division)]["W"].nlargest(division_winners).min()
        )
    # print(f'Win thresholds by division: {division_win_threshold}')
    def i_fail_at_pandas(division):
        return division_win_threshold[division]

    df["is_contender"] = df["max_wins"] >= df["div"].apply(i_fail_at_pandas)
    output = df.loc[df["is_contender"]]["div"]
    # print(f'Here is what I think I will return: {output}')
    return output


def retro_to_datetime(retro_str):
    return datetime.strptime(retro_str, "%Y%m%d")


def datetime_to_retro(dt):
    return datetime.strftime(dt, "%Y%m%d")


def load_schedule(filename):
    schedule_columns = [
        "date",
        "doubleheader_index",
        "weekday",
        "visitor",
        "visitor_league",
        "visitor_game_num",
        "home",
        "home_league",
        "home_game_num",
        "day_night",
        "completion",
        "makeup_date",
    ]
    return pd.read_csv(
        filename,
        header=None,
        names=schedule_columns,
        usecols=range(len(schedule_columns)),
    )


#    df['completion'] = df['completion'].astype(str)
#    df['makeup_date'] = df['makeup_date'].astype(str)
#    return df.dropna(subset=['home'])

schedule_path = "./data/2021SKED.TXT"
SCHEDULE = load_schedule(schedule_path)


def find_unplayed_games(schedule):
    return schedule.loc[
        (schedule["makeup_date"].str.startswith("not", na=False))
        | (schedule["completion"].str.contains("No makeup", na=False))
    ]


def logged_games_after_date(df, date):
    return df.loc[df["completion_date"] > date]


def all_matchups_after_date(played, unplayed, date):
    logged_games = logged_games_after_date(played, date)[
        ["completion_date", "visitor", "home"]
    ]
    unplayed_games = unplayed
    all_games = pd.concat([logged_games, unplayed_games], ignore_index=True)
    alpha_pairs = pd.DataFrame(
        np.where(
            all_games["visitor"] < all_games["home"],
            (all_games["visitor"], all_games["home"]),
            (all_games["home"], all_games["visitor"]),
        )
    )
    return alpha_pairs.T.rename(columns={0: "alpha1", 1: "alpha2"})


def divisional_threats(standings_immutable, season_params: SeasonParameters, team: str):
    # We only care about teams with more max_wins than us.
    df = standings_immutable.copy()
    my_division = season_params.divisions.loc[team]["div"]
    my_league = season_params.divisions.loc[team]["lg"]
    df = df.merge(season_params.divisions, left_index=True, right_index=True).loc[
        season_params.divisions["div"] == my_division
    ]
    max_wins = season_params.season_lengths[my_league] - df["L"]
    df["max_wins"] = max_wins  # please kill me
    # print(f'max_wins: {max_wins}')
    df = df.loc[(df.index != team) & (max_wins[df.index] >= max_wins[team])]
    # print(f'After that df.loc business: {df}')
    # So now we have all the teams with more max wins than us.
    # We don't care about the top n-1 where winners_per_division is n.
    # I don't think it's sorted yet.
    df = df.sort_values(by=["max_wins"], ascending=False).iloc[
        (season_params.winners_per_division - 1) :
    ]
    # Actually using the head to head records is coming, I swear.
    if season_params.tiebreakers_required:
        return max_wins[team] - df["W"] - 1
    else:
        return max_wins[team] - df["W"]


def get_division_contenders2(played, unplayed, season_params: SeasonParameters, date):
    standings = compute_standings(played)
    gms_played = game_log.copy()
    # We'll have to handle ties and forfeits elsewhere for Retrosheet.
    winners = pd.Series(
        np.where(gms_played["home_won"], gms_played["home"], gms_played["visitor"])
    )
    losers = pd.Series(
        np.where(gms_played["home_won"], gms_played["visitor"], gms_played["home"])
    )
    standings = pd.concat(
        [winners.value_counts().rename("W"), losers.value_counts().rename("L")], axis=1
    )
    return standings.fillna(0)

    # We only care about teams with more max_wins than us.
    df = standings_immutable.copy()
    my_division = season_params.divisions.loc[team]["div"]
    my_league = season_params.divisions.loc[team]["lg"]
    df = df.merge(season_params.divisions, left_index=True, right_index=True).loc[
        season_params.divisions["div"] == my_division
    ]
    max_wins = season_length[my_league] - df["L"]
    df["max_wins"] = max_wins  # please kill me
    df = df.loc[(df.index != team) & (max_wins[df.index] >= max_wins[team])]
    # print(f'After that df.loc business: {df}')
    # So now we have all the teams with more max wins than us.
    # We don't care about the top n-1 where winners_per_division is n.
    # I don't think it's sorted yet.
    df = df.sort_values(by=["max_wins"], ascending=False).iloc[
        (season_params.winners_per_division - 1) :
    ]
    return max_wins[team] - df["W"] - 1


def games_between_rivals_after_date(played, unplayed, season_params, date, team):
    remaining = (
        all_matchups_after_date(played, unplayed, date)
        .groupby(["alpha1", "alpha2"], as_index=False)
        .size()
    )
    standings = compute_standings(played.loc[(played["completion_date"] <= date)])
    threats = divisional_threats(standings, season_params, team)
    return remaining.loc[
        (remaining["alpha1"].isin(threats.index))
        & (remaining["alpha2"].isin(threats.index))
    ]


# I thought there was one season length per season but it's really per league
# per season.
def get_season_lengths(schedule):
    home = schedule[["home", "home_league"]].rename(
        columns={"home": "team", "home_league": "league"}
    )
    visitors = schedule[["visitor", "visitor_league"]].rename(
        columns={"visitor": "team", "visitor_league": "league"}
    )
    # if pd.concat([home, visitors]).value_counts().groupby("league").nunique().eq(1)
    grouped = pd.concat([home, visitors]).value_counts().groupby("league")
    if grouped.nunique().eq(1).all():
        return grouped.min()
    else:
        print(pd.concat([home, visitors]).value_counts())
        assert grouped.min() == grouped.max()


def get_season_lengths_statsapi(played, unplayed, divisions):
    schedule = pd.concat([played, unplayed], ignore_index=True)
    schedule = schedule.merge(divisions, left_on="home", right_index=True).rename(
        columns={"lg": "home_league"}
    )
    schedule = schedule[["home", "visitor", "home_league"]]
    schedule = schedule.merge(divisions, left_on="visitor", right_index=True).rename(
        columns={"lg": "visitor_league"}
    )[["home", "visitor", "home_league", "visitor_league"]]
    return get_season_lengths(schedule)


# sort the output of division_threats
def sort_rivals(matchups, threats):
    # print(matchups.columns, threats.shape)
    df = matchups.copy()
    merge1 = df.merge(threats, left_on=["alpha1"], right_index=True)
    # print(f'now merge1 is: {merge1}')
    merge2 = merge1.merge(threats, left_on=["alpha2"], right_index=True)
    # print(f'now merge2 is: {merge2}')
    merge2["betterT"] = np.where(
        merge2["W_x"] >= merge2["W_y"], merge2["alpha2"], merge2["alpha1"]
    )
    merge2["worseT"] = np.where(
        merge2["W_x"] >= merge2["W_y"], merge2["alpha1"], merge2["alpha2"]
    )
    merge2["lesserW"] = merge2[["W_x", "W_y"]].min(axis=1)
    merge2["greaterW"] = merge2[["W_x", "W_y"]].max(axis=1)
    return merge2.sort_values(["lesserW", "greaterW"], ascending=True)[
        ["betterT", "worseT", "size"]
    ]


# This never worked reliably. I'm leaving it here for historical reasons in
# case someone ever wants to revisit this approach.
def portion_out_wins(sorted_rivals, threats):
    for _, row in sorted_rivals.iterrows():
        games = row["size"]
        betterT = row["betterT"]
        worseT = row["worseT"]
        threat1, threat2 = threats[betterT], threats[worseT]
        print(
            f"We can afford to let {betterT} win {threat1} more games, and {worseT} {threat2}"
        )
        if games > threat1 + threat2:
            # No matter what, these games will lead to too many wins for one team.
            print(f"But there are {games} games left between {betterT} and {worseT}")
            return False
        # We try to give as many wins to the best team as possible first. Then distribute them
        # downward. Not sure about this.
        # In the Baltimore example, Tampa Bay cannot win another game. So wins_for_t1 is going to be
        # zero. We give all three of the TB-NY series to NY.
        wins_for_t1 = min(threat1, games)
        wins_for_t2 = games - wins_for_t1
        print(
            f"Let's say that {betterT} wins {wins_for_t1} of their remaining {games} with {worseT}"
        )
        threats[row["betterT"]] -= wins_for_t1
        threats[row["worseT"]] -= wins_for_t2
    return True


def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


def all_subset_sums(matchups, threats, team_str="This team"):
    all_rivals = threats.index
    for subset in powerset(all_rivals):
        if len(subset) < 2:
            continue
        if "betterT" in matchups.columns:
            team1, team2 = "betterT", "worseT"
        else:
            team1, team2 = "alpha1", "alpha2"
        subset_matchup_count = matchups.loc[
            (matchups[team1].isin(subset)) & (matchups[team2].isin(subset))
        ]["size"].sum()
        allowable_win_count = threats.loc[threats.index.isin(subset)].sum()
        offset = allowable_win_count - subset_matchup_count
        if subset_matchup_count <= allowable_win_count:
            if offset < 30:
                # print(f'We can allow {subset} to win {allowable_win_count} and they only have {subset_matchup_count} games left so that seems okay, but the elimination number is around {offset + 1}')
                pass
            pass
        else:
            print(
                f"{team_str} can only allow {subset} to win {allowable_win_count} total games but they have {subset_matchup_count} games left against each other."
            )
            return False
    return True


def is_division_contender_with_rivalries(
    played, unplayed, season_params: SeasonParameters, date, team: str
):
    matchups = games_between_rivals_after_date(
        played, unplayed, season_params, date, team
    )
    threats = divisional_threats(
        compute_standings(played.loc[(played["completion_date"] <= date)]),
        season_params,
        team,
    )
    # print(f'threats: {threats}')
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats, team)


def wildcard_standings(standings_immutable, season_params):
    contenders = set()
    if not season_params.wildcard_count:
        return pd.DataFrame()
    df = standings_immutable.copy()
    # I hate myself for this
    df["div"] = season_params.divisions["div"]
    df["lg"] = season_params.divisions["lg"]
    # I have no idea what I am doing.
    def why_is_this_necessary(league):
        return season_params.season_lengths[league]

    df["season_length"] = df["lg"].apply(why_is_this_necessary)

    df["max_wins"] = df["season_length"] - df["L"]

    df["division_leader"] = False
    df = df.sort_values(by=["max_wins"], ascending=False)
    df.loc[
        df.groupby("div").head(season_params.winners_per_division).index,
        "division_leader",
    ] = True
    # print(f"df after loc.groupby business: {df}")
    wildcard_wins_by_league = (
        df.loc[df["division_leader"] == False]
        .sort_values(by=["W"], ascending=False)
        .groupby("lg")
        .nth(season_params.wildcard_count - 1)["W"]
    )
    # print(f"wildcard_wins_by_league: {wildcard_wins_by_league}")
    merge1 = df.merge(wildcard_wins_by_league, left_on=["lg"], right_index=True)
    # print(f"merge1: {merge1}")
    # print(f'merge1 OAK: {merge1.loc["OAK"]}')
    return merge1.loc[merge1["max_wins"] >= merge1["W_y"]][
        ["W_x", "L", "div", "lg", "max_wins", "division_leader"]
    ].rename(columns={"W_x": "W"})
    # return merge1[["W_x", "L", "div", "lg", "max_wins", "division_leader"]].rename(
    #    columns={"W_x": "W"}
    # )


def wildcard_contenders_naive(standings_immutable, season_params):
    # standings = wildcard_standings(standings_immutable, season_params)
    return set(wildcard_standings(standings_immutable, season_params).index)


def wildcard_threats(standings_immutable, season_params, team):
    # We only care about teams with more max_wins than us
    df = standings_immutable.copy()
    # this is so bad
    df["lg"] = season_params.divisions["lg"]

    my_league = season_params.divisions.loc[team]["lg"]
    my_max_wins = df.loc[team]["max_wins"]
    df = df.loc[(df["lg"] == my_league) & (df["division_leader"] == False)]
    df = df.drop(df.nlargest(season_params.wildcard_count - 1, columns=["W"]).index)
    return my_max_wins - df["W"]


def games_between_wildcard_rivals_after_date(
    played, unplayed, season_params: SeasonParameters, date, team
):
    remaining = (
        all_matchups_after_date(played, unplayed, date)
        .groupby(["alpha1", "alpha2"], as_index=False)
        .size()
    )
    basic_standings = compute_standings(played.loc[(played["completion_date"] <= date)])
    standings = wildcard_standings(
        compute_standings(played.loc[(played["completion_date"] <= date)]),
        season_params,
    )
    threats = wildcard_threats(standings, season_params, team)
    # print(f'wildcard threats: {threats}')
    return remaining.loc[
        (remaining["alpha1"].isin(threats.index))
        & (remaining["alpha2"].isin(threats.index))
    ]


def is_wild_card_contender_with_rivalries(played, unplayed, season_params, date, team):
    matchups = games_between_wildcard_rivals_after_date(
        played, unplayed, season_params, date, team
    )
    # print(f'matchups: {matchups}')
    standings = wildcard_standings(
        compute_standings(played.loc[(played["completion_date"] <= date)]),
        season_params,
    )
    # print(f'About to call wildcard_threats where standings has these columns: {standings.columns}')
    threats = wildcard_threats(standings, season_params, team)
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats, team)


def count_teams(played, unplayed, divisions):
    schedule_teams = sorted(
        pd.unique(pd.concat([played, unplayed])[["home", "visitor"]].values.ravel("K"))
    )
    divisions_count = len(divisions.index)
    # TODO: When we convert retro game log and schedule into played/unplayed,
    # we need to check that schedule/GL teams match.
    if any(t not in divisions.index for t in schedule_teams):
        print(f"Teams in schedule: {schedule_teams} ")
        print(f"Teams in division map: {sorted(divisions.index)}")
    assert all(t in divisions.index for t in schedule_teams)
    return len(schedule_teams)


def retrosheet_to_played_unplayed(game_log, schedule, season_params: SeasonParameters):
    unplayed = schedule.copy()
    unplayed["completion"] = unplayed["completion"].astype(str)
    unplayed["makeup_date"] = unplayed["makeup_date"].astype(str)
    unplayed = unplayed.loc[
        (unplayed["makeup_date"].str.startswith("not", na=False))
        | (unplayed["completion"].str.contains("No makeup", na=False))
    ]
    unplayed = unplayed[["home", "visitor"]]
    unplayed["completion_date"] = np.nan  # I'm going to regret this.
    # Do we still need to dropna(subset=['home']) on unplayed?
    played = game_log.copy()
    played = played.drop(
        played[played["home_game_num"] > season_params.season_lengths.max()].index
    )

    played["home_won"] = False
    played.loc[
        (played["home_score"] > played["visitor_score"]) | (played["forfeit"] == "H"),
        "home_won",
    ] = True
    played["date"] = pd.to_datetime(played["date"], format="%Y%m%d")
    # delete ties but keep forfeits for now
    played = played[
        (played["visitor_score"] != played["home_score"]) | (played["forfeit"].notna())
    ]
    # delete forfeits scored as ties
    played = played[played["forfeit"] != "T"]
    played = played.fillna(value={"completion": "17760704,I hate Pandas"})
    # Now I can ensure that the completion column is all strings.
    played["completion"] = played["completion"].astype(str)
    played["completion_date"] = played["date"]
    # If there is anything in completion, extract the date and put that in completion_date
    # return played.loc[(played['completion'] != "17760704,I hate Pandas")][['completion']]
    played.loc[
        (played["completion"] != "17760704,I hate Pandas"), "completion_date"
    ] = pd.to_datetime(
        played["completion"].str.split(",").str[0], format="%Y%m%d"
    )  # needs to be datetime!
    played = played[["completion_date", "home", "visitor", "home_won"]]
    return played, unplayed


def display_name(divisions, team_id):
    team_entry = divisions.loc[team_id]
    return f'{team_entry["location"]} {team_entry["nickname"]}'


def show_dumb_elimination_output3(
    df, schedule, divisions, wildcard_count=2, winners_per_division=1
):
    team_count = count_teams(df, schedule, divisions)
    games_per_season = get_season_length(schedule)
    print(f"DO NOT RUN THIS CODE: This season has {team_count} teams.")
    print(
        f"The top {winners_per_division} teams from each division go to the postseason."
    )
    for index, value in games_per_season.iteritems():
        print(f"The {index} has {value} games per team.")
    # If home_game_num > games_per_season this is probably a tiebreaker playoff.
    # I guess each league could have their tiebreakers on different days? I hope
    # this doesn't exclude a regular season game. Maybe I should make the check
    # in compute_standings.
    max_date = retro_to_datetime(
        df.loc[df["home_game_num"] <= games_per_season.max()]["completion_date"].max()
    )
    print(f"max_date is {max_date}")
    min_date = retro_to_datetime(df["completion_date"].min())

    current_date = max_date
    div_contenders = set()
    tomorrows_div_contenders = None
    wildcard_contenders = set()
    tomorrows_wildcard_contenders = None
    tomorrows_contenders_any = None
    tomorrows_standings = None
    eliminations = {}
    while current_date > min_date and (
        (len(div_contenders) < team_count)
        or wildcard_count
        and len(wildcard_contenders) < team_count
    ):
        date_str = datetime_to_retro(current_date)
        # print(f'Starting analysis of {date_str}')
        current_standings = compute_standings(df.loc[df["completion_date"] <= date_str])
        # print(current_standings)
        # MOVE THIS SHIT INTO SOMETHING EFFICIENT
        current_standings["div"] = divisions["div"]
        current_standings["lg"] = divisions["lg"]
        # I have no idea what I am doing.
        def why_is_this_necessary(league):
            return games_per_season[league]

        current_standings["season_length"] = current_standings["lg"].apply(
            why_is_this_necessary
        )
        current_standings["max_wins"] = (
            current_standings["season_length"] - current_standings["L"]
        )
        # print(current_standings)

        div_contenders = division_contenders(
            current_standings, games_per_season, division_winners=winners_per_division
        )
        # print(f'naive division contenders: {div_contenders}')
        for supposed_contender in div_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if (
                tomorrows_div_contenders
                and supposed_contender not in tomorrows_div_contenders
            ):
                if not is_division_contender_with_rivalries(
                    df,
                    schedule,
                    divisions,
                    winners_per_division=winners_per_division,
                    date_str=date_str,
                    team=supposed_contender,
                ):
                    print(
                        f"It looked like the {display_name(divisions, supposed_contender)} were in contention after {date_str} but the remaining intra-division games ruled them out."
                    )
                    div_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_div_contenders:
            new_contenders = div_contenders.difference(tomorrows_div_contenders)
        # if new_contenders:
        # print(f'Teams eliminated from their division titles on {datetime_to_retro(tomorrow)}: {new_contenders}')
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = (
                tomorrows_standings.loc[eliminated_team]["season_length"]
                - tomorrows_standings.loc[eliminated_team]["W"]
                - tomorrows_standings.loc[eliminated_team]["L"]
            )
            elim_div = divisions.loc[eliminated_team]["div"]
            print(
                f"The {display_name(divisions, eliminated_team)} were eliminated from the {elim_div} title on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play."
            )
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = divisions.loc[eliminated_team]["franchise_id"]
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]["division"] = new_pair
            else:
                eliminations[elim_franchise] = {"division": new_pair}
        # print(f'PHI max wins: {current_standings.loc["PHI"]["max_wins"]}, SLN wins: {current_standings.loc["SLN"]["W"]}')
        # print(f'My busted view of the wildcard standings: {wildcard_standings(current_standings, divisions, wildcard_count, division_winners=winners_per_division, games_per_season=games_per_season)}')
        wildcard_contenders = wildcard_contenders_naive(
            current_standings,
            divisions,
            wildcard_count,
            winners_per_division,
            games_per_season,
        )
        # print(f'naive wildcard contenders after {date_str} games: {wildcard_contenders}')
        for supposed_contender in wildcard_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if (
                tomorrows_wildcard_contenders
                and supposed_contender not in tomorrows_wildcard_contenders
            ):
                # print(f'let\'s see if {supposed_contender} is really still in wildcard contention')
                if not is_wildcard_contender_with_rivalries(
                    df,
                    schedule,
                    divisions,
                    date_str,
                    supposed_contender,
                    wildcard_count,
                    winners_per_division,
                ):
                    print(
                        f"It looked like the {display_name(divisions, supposed_contender)} were in wildcard contention after {date_str} but the remaining intra-contender games ruled them out."
                    )
                    wildcard_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_wildcard_contenders:
            new_contenders = wildcard_contenders.difference(
                tomorrows_wildcard_contenders
            )
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = (
                tomorrows_standings.loc[eliminated_team]["season_length"]
                - tomorrows_standings.loc[eliminated_team]["W"]
                - tomorrows_standings.loc[eliminated_team]["L"]
            )
            elim_lg = divisions.loc[eliminated_team]["lg"]

            print(
                f"The {display_name(divisions, eliminated_team)} were eliminated from {elim_lg} wildcard contention on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play."
            )
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = divisions.loc[eliminated_team]["franchise_id"]
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]["wildcard"] = new_pair
            else:
                eliminations[elim_franchise] = {"wildcard": new_pair}

        contenders_any = div_contenders.union(wildcard_contenders)
        if tomorrows_contenders_any:
            new_contenders = contenders_any.difference(tomorrows_contenders_any)
        # if new_contenders:
        #     print(f'Teams eliminated from ALL postseason contention on {datetime_to_retro(tomorrow)}: {new_contenders}')
        tomorrow = current_date
        tomorrows_div_contenders = div_contenders.copy()
        tomorrows_wildcard_contenders = wildcard_contenders.copy()
        tomorrows_contenders_any = contenders_any.copy()
        tomorrows_standings = current_standings.copy()
        current_date = current_date - timedelta(days=1)
    return eliminations


def wildcards_for_year(year):
    if year >= 2022:
        return 3
    if year >= 2012:
        return 2
    # I thought I had to special-case 1994 but really it just (correctly)
    # outputs that everyone was in contention on the last day so there are no
    # eliminations.
    if year >= 1994:
        return 1
    if year == 1981:
        return "1981 was weird sorry"
    return 0


def get_winners_per_division(year):
    if year == 2020:
        return 2
    return 1


def run_one_year_retro(year):
    # print(f"starting analysis of {year}")
    nicknames = load_nicknames("data/CurrentNames.csv")
    team_ids = load_team_ids("data/TEAMABR.TXT")
    game_log = load_game_log(f"./data/GL{year}.TXT")
    if year == 2020:
        schedule = load_schedule("./data/2020REV.TXT")
    else:
        schedule = load_schedule(f"./data/{year}SKED.TXT")
    if year == 1880:
        schedule = fix_1880(schedule)
    if year == 1882:
        game_log = fix_1882(game_log)
    if year == 1884:
        game_log, schedule = fix_1884(game_log, schedule)
    if year == 1890:
        game_log, schedule = fix_1890(game_log, schedule)
    if year == 1891:
        game_log, schedule = fix_1891(game_log, schedule)
    if year == 1905:
        print(
            "I am not handling ties correctly, so this output is going to imply Chicago was still in contention at the end of the season. Sorry, I hope to fix it soon!"
        )
    season_params = SeasonParameters(year, nicknames, team_ids, schedule)
    played, unplayed = retrosheet_to_played_unplayed(game_log, schedule, season_params)
    return show_dumb_elimination_output4(played, unplayed, season_params)


# In[50]:


# test case - the Mets after 1964-08-14
# and every date up to August 28 - they are obviously out on the 29th


def assign_wins_with_brute_force(
    sorted_rivals, threats_immutable, divisions, recursion_level=0
):
    def myprint(mytext):
        if recursion_level <= 3:
            print(" " * 2 * recursion_level, mytext)

    threats = threats_immutable.copy()
    if sorted_rivals.empty:
        return {}
    #    if all_subset_sums(sorted_rivals.rename(columns={'betterT': 'alpha1', 'worseT': 'alpha2'}), threats):
    #        pass
    #    else:
    #        return None

    row = sorted_rivals.iloc[0]
    remaining_rows = sorted_rivals.iloc[1:, :]
    games = row["size"]
    betterT = row["betterT"]
    worseT = row["worseT"]
    threat1, threat2 = threats[betterT], threats[worseT]
    myprint(
        f"We can afford to let the {display_name(divisions, betterT)} win {threat1} more games, and the {display_name(divisions, worseT)} {threat2}"
    )
    if threat1 < 0 or threat2 < 0:
        myprint(f"How the fuck did that get negative?")
        return None
    if games > threat1 + threat2:
        # No matter what, these games will lead to too many wins for one team.
        myprint(
            f"But there are {games} games left between the {display_name(divisions, betterT)} and the {display_name(divisions, worseT)}"
        )
        return None
    minWinsForT1 = max(0, games - threat2)
    maxWinsForT1 = min(threat1, games)
    myprint(
        f"The {display_name(divisions, betterT)} have to win between {minWinsForT1} and {maxWinsForT1} of these {games} games."
    )
    for winsForT1 in range(minWinsForT1, maxWinsForT1 + 1):
        winsForT2 = games - winsForT1
        myprint(
            f"What if the {display_name(divisions, betterT)} won {winsForT1} out of {games} and the {display_name(divisions, worseT)} won {winsForT2}?"
        )
        if winsForT1 > threat1:
            myprint(
                f"That is too many wins for the {display_name(divisions, betterT)} so we're done here."
            )
            return None
        if winsForT2 > threat2:
            myprint(f"That is too many wins for the {display_name(divisions, worseT)}.")
            continue
        # get ready to recurse
        new_threats = threats.copy()
        new_threats[betterT] -= winsForT1
        new_threats[worseT] -= winsForT2
        recurse = assign_wins_with_brute_force(
            remaining_rows, new_threats, divisions, recursion_level + 1
        )
        if recurse != None:
            recurse[(betterT, worseT)] = winsForT1
            return recurse
        # If that recursive check was false, we'll keep going on the for loop.
    return None


def check_division_contention(date_str, year, team):
    game_log = load_game_log(f"./data/GL{year}.TXT")
    schedule = load_schedule(f"./data/{year}SKED.TXT")
    divisions = divisions_for_year(NICKNAMES, TEAM_IDS_UNDEFINED_LOL, year)
    season_lengths = get_season_length(schedule)
    matchups = games_between_rivals_after_date(
        game_log, schedule, divisions, date_str, team
    )
    threats = divisional_threats(
        compute_standings(game_log.loc[(game_log["completion_date"] <= date_str)]),
        divisions,
        season_lengths,
        team,
    )
    sum_mode = dumb_matrix_sum(matchups, threats)
    print(f"Sum mode says {sum_mode}")
    # sum mode errs on the side of True. If it's False, brute force will never return True.
    if not sum_mode:
        return False
    subset_mode = all_subset_sums(matchups, threats)
    print(f"Subset mode says {subset_mode}")
    # subset mode errs on the side of True. If it's False, brute force will never return True.
    if not subset_mode:
        return False
    sorted_rivals = sort_rivals(matchups, threats)
    easy_mode = is_division_contender_with_rivalries(
        game_log, schedule, divisions, winners_per_division, date_str, team
    )
    # print(f'Easy mode says {easy_mode}')
    # Easy mode errs on the side of False. If it's true, brute force will never return False.
    # if easy_mode:
    #    return True
    brute_force = brute_force_rival_matchups(sorted_rivals, threats)
    print("brute force completed")
    return brute_force


def dumb_matrix_sum(matchups, threats):
    total_matchups = matchups["size"].sum()
    total_allowable_wins = threats.sum()
    return total_allowable_wins >= total_matchups


# they both say false for 19640823
# brute force says true for 19640814
# check_division_contention('20210821', 2021, 'BAL')

# NICKNAMES = load_nicknames('data/CurrentNames.csv')
# run_one_year(1899)
correct_1899_standings = '{"W":{"BRO":101,"BSN":95,"PHI":94,"BLN":86,"SLN":84,"CIN":83,"PIT":76,"CHN":75,"LS3":75,"NY1":60,"WSN":54,"CL4":20},"L":{"BRO":47,"BSN":57,"PHI":58,"BLN":62,"SLN":67,"CIN":67,"PIT":73,"CHN":73,"LS3":77,"NY1":90,"WSN":98,"CL4":134}}'
# thanks https://stackoverflow.com/a/57915246
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# print(run_one_year(2020))
# for year in [2021]:
#    output = run_one_year(year)
#    with open(f'output/{year}.json', 'w') as fp:
#      json.dump(output, fp, cls=NpEncoder)

ELO_TO_RETRO = {
    "CHC": "CHN",
    "CHW": "CHA",
    "FLA": "MIA",
    "KCR": "KCA",
    "LAD": "LAN",
    "NYM": "NYN",
    "NYY": "NYA",
    "SDP": "SDN",
    "SFG": "SFN",
    "STL": "SLN",
    "TBD": "TBA",
    "WSN": "WAS",
}


def elo_to_played_and_unplayed(elo_immutable):
    elo = elo_immutable.replace({"team1": ELO_TO_RETRO, "team2": ELO_TO_RETRO})
    elo = elo.rename(
        columns={
            "team1": "home",
            "team2": "visitor",
            "date": "completion_date",
            "score1": "home_score",
            "score2": "visitor_score",
        }
    )
    elo["completion_date"] = pd.to_datetime(elo["completion_date"])
    unplayed = elo[elo["home_score"].isnull()][["visitor", "home", "completion_date"]]
    # What if we didn't care about the runs? What if we just calculated winner and loser now?
    played = elo.copy().dropna(subset="home_score")
    played["home_won"] = False
    played.loc[played["home_score"] > played["visitor_score"], "home_won"] = True
    played = played[["visitor", "home", "completion_date", "home_won"]]
    return played, unplayed


def statsapi_schedule_to_played_unplayed(schedule_json_path):
    played_tuples = []
    unplayed_tuples = []

    full_schedule = json.load(open(schedule_json_path))
    DEBUG_TEAM = "Carolina Mudcats"
    DEBUG_GAMES = 0
    for date in full_schedule["dates"]:
        for game in date["games"]:
            if game["gameType"] != "R":
                continue
            date_str = game["officialDate"]
            date_pd = pd.to_datetime(date_str)
            home = game["teams"]["home"]["team"]["name"]
            visitor = game["teams"]["away"]["team"]["name"]
            # print(f'Trying to figure out what to do with {date_str} {visitor}@{home} {game["status"]}')
            if game.get("resumeDate"):
                print(
                    f"Skipping the {date_str} {visitor}@{home} game because it has resumeDate."
                )
                continue
            if game["status"]["codedGameState"] in ["F", "O"]:
                if game["teams"]["home"]["isWinner"]:
                    home_won = True
                elif game["teams"]["away"]["isWinner"]:
                    home_won = False
                else:
                    raise (f"No winner, that\s messed up.")
                new_tuple = (date_pd, home, visitor, home_won)
                played_tuples.append(new_tuple)
                if DEBUG_TEAM in [home, visitor]:
                    DEBUG_GAMES += 1
                    print(f"The {DEBUG_TEAM} have now played {DEBUG_GAMES} games.")

            elif game["status"]["codedGameState"] in ["S", "I", "P", "C", "U"]:
                unplayed_tuples.append((date_pd, home, visitor))
                if DEBUG_TEAM in [home, visitor]:
                    DEBUG_GAMES += 1
                    print(f"The {DEBUG_TEAM} have now played {DEBUG_GAMES} games.")
            else:
                # print(f'I don\'t know what to do with codedGameState {game["status"]["codedGameState"]} for the {date_str} {visitor}@{home} game.')
                pass
    played = pd.DataFrame.from_records(
        np.array(
            played_tuples,
            dtype=[
                ("completion_date", "datetime64[us]"),
                ("home", "O"),
                ("visitor", "O"),
                ("home_won", "bool"),
            ],
        )
    )
    unplayed = pd.DataFrame.from_records(
        np.array(
            unplayed_tuples,
            dtype=[
                ("completion_date", "datetime64[us]"),
                ("home", "O"),
                ("visitor", "O"),
            ],
        )
    )

    return played, unplayed


def show_dumb_elimination_output4(played, unplayed, season_params):
    team_count = count_teams(played, unplayed, season_params.divisions)
    print(f"This season has {team_count} teams.")
    print(
        f"The top {season_params.winners_per_division} teams from each division go to the postseason, plus {season_params.wildcard_count} wild cards."
    )
    for index, value in season_params.season_lengths.iteritems():
        print(f"The {index} has {value} games per team.")
    # We don't have home_game_num in the ELO game log. We need to filter that
    # out of the Retrosheet game log somewhere else. Or else add it to the ELO
    # game log, but I hope that isn't necessary.
    max_date = played["completion_date"].max()
    # print(f"max_date is {max_date}")
    min_date = played["completion_date"].min()

    current_date = max_date
    div_contenders = set()
    tomorrows_div_contenders = None
    wildcard_contenders = set()
    tomorrows_wildcard_contenders = None
    tomorrows_contenders_any = None
    tomorrows_standings = None
    eliminations = {}
    while current_date > min_date and (
        (len(div_contenders) < team_count)
        or season_params.wildcard_count
        and len(wildcard_contenders) < team_count
    ):
        date_str = datetime_to_retro(current_date)
        # print(f"Starting analysis of {date_str}")
        current_standings = compute_standings(
            played.loc[played["completion_date"] <= date_str]
        )
        # print(current_standings)
        # MOVE THIS SHIT INTO SOMETHING EFFICIENT
        current_standings["div"] = season_params.divisions["div"]
        current_standings["lg"] = season_params.divisions["lg"]
        # I have no idea what I am doing.
        def why_is_this_necessary(league):
            return season_params.season_lengths[league]

        current_standings["season_length"] = current_standings["lg"].apply(
            why_is_this_necessary
        )
        current_standings["max_wins"] = (
            current_standings["season_length"] - current_standings["L"]
        )
        # print(current_standings)

        div_contenders_df = division_contenders(
            current_standings,
            season_params.season_lengths,
            division_winners=season_params.winners_per_division,
        )
        if current_date == max_date:
            # check for end-of-season ties
            # print(f"div_contenders_df: {div_contenders_df}")
            winners_by_division = div_contenders_df.groupby(div_contenders_df).size()
            # print(f"winners_by_division: {winners_by_division}")
            for (index, value) in winners_by_division.items():
                if value > season_params.winners_per_division:
                    contenders_set = set(
                        div_contenders_df.loc[lambda x: x == index].index
                    )
                    print(
                        f"The {index} has more contenders at the end of the season than I expected: {contenders_set}. This discrepancy is either because I still handle ties incorrectly, or it was an actual tie and they held a playoff later on."
                    )

        div_contenders = set(div_contenders_df.index)
        # print(f"naive division contenders: {sorted(div_contenders)}")
        for supposed_contender in sorted(div_contenders.copy()):
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if (
                tomorrows_div_contenders
                and supposed_contender not in tomorrows_div_contenders
            ):
                if not is_division_contender_with_rivalries(
                    played, unplayed, season_params, current_date, supposed_contender
                ):
                    print(
                        f"It looked like the {display_name(season_params.divisions, supposed_contender)} were in contention after {date_str} but the remaining intra-division games ruled them out."
                    )
                    div_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_div_contenders:
            new_contenders = div_contenders.difference(tomorrows_div_contenders)
        # if new_contenders:
        # print(f'Teams eliminated from their division titles on {datetime_to_retro(tomorrow)}: {new_contenders}')
        for eliminated_team in sorted(new_contenders):
            games_to_go_at_elimination = (
                tomorrows_standings.loc[eliminated_team]["season_length"]
                - tomorrows_standings.loc[eliminated_team]["W"]
                - tomorrows_standings.loc[eliminated_team]["L"]
            )
            elim_div = season_params.divisions.loc[eliminated_team]["div"]
            print(
                f"The {display_name(season_params.divisions, eliminated_team)} were eliminated from the {elim_div} title on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play."
            )
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = season_params.divisions.loc[eliminated_team][
                "franchise_id"
            ]
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]["division"] = new_pair
            else:
                eliminations[elim_franchise] = {"division": new_pair}
        # print(f'PHI max wins: {current_standings.loc["PHI"]["max_wins"]}, SLN wins: {current_standings.loc["SLN"]["W"]}')
        # print(f'My busted view of the wildcard standings: {wildcard_standings(current_standings, divisions, wildcard_count, division_winners=winners_per_division, games_per_season=games_per_season)}')
        wildcard_contenders = wildcard_contenders_naive(
            current_standings, season_params
        )
        # print(
        #   f"naive wildcard contenders after {date_str} games: {sorted(wildcard_contenders)}"
        # )
        for supposed_contender in sorted(wildcard_contenders.copy()):
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if (
                tomorrows_wildcard_contenders
                and supposed_contender not in tomorrows_wildcard_contenders
            ):
                # print(f'let\'s see if {supposed_contender} is really still in wildcard contention')
                if not is_wild_card_contender_with_rivalries(
                    played, unplayed, season_params, current_date, supposed_contender
                ):
                    print(
                        f"It looked like the {display_name(season_params.divisions, supposed_contender)} were in wildcard contention after {date_str} but the remaining intra-contender games ruled them out."
                    )
                    wildcard_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_wildcard_contenders:
            new_contenders = wildcard_contenders.difference(
                tomorrows_wildcard_contenders
            )
        for eliminated_team in sorted(new_contenders):
            games_to_go_at_elimination = (
                tomorrows_standings.loc[eliminated_team]["season_length"]
                - tomorrows_standings.loc[eliminated_team]["W"]
                - tomorrows_standings.loc[eliminated_team]["L"]
            )
            elim_lg = season_params.divisions.loc[eliminated_team]["lg"]

            print(
                f"The {display_name(season_params.divisions, eliminated_team)} were eliminated from {elim_lg} wildcard contention on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play."
            )
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = season_params.divisions.loc[eliminated_team][
                "franchise_id"
            ]
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]["wildcard"] = new_pair
            else:
                eliminations[elim_franchise] = {"wildcard": new_pair}

        contenders_any = div_contenders.union(wildcard_contenders)
        if tomorrows_contenders_any:
            new_contenders = contenders_any.difference(tomorrows_contenders_any)
        # if new_contenders:
        #     print(f'Teams eliminated from ALL postseason contention on {datetime_to_retro(tomorrow)}: {new_contenders}')
        tomorrow = current_date
        tomorrows_div_contenders = div_contenders.copy()
        tomorrows_wildcard_contenders = wildcard_contenders.copy()
        tomorrows_contenders_any = contenders_any.copy()
        tomorrows_standings = current_standings.copy()
        current_date = current_date - timedelta(days=1)
    return eliminations


def head_to_head_records(played):
    def alpha1_won(r):
        output = {}
        output["alpha1"] = min(r["home"], r["visitor"])
        output["alpha2"] = max(r["home"], r["visitor"])
        if (r["home"] < r["visitor"] and r["home_won"]) or (
            r["home"] > r["visitor"] and (not r["home_won"])
        ):
            output["alpha1_wins"], output["alpha2_wins"] = (1, 0)
        else:
            output["alpha1_wins"], output["alpha2_wins"] = (0, 1)
        return pd.Series(output)

    matchups_alpha = pd.DataFrame(played.apply(alpha1_won, axis=1))
    return matchups_alpha.groupby(["alpha1", "alpha2"]).sum()


def get_division_rivals(standings, divisions, team):
    my_division = divisions.loc[team]["div"]
    my_league = divisions.loc[team]["lg"]
    df = standings.merge(divisions, left_index=True, right_index=True).loc[
        divisions["div"] == my_division
    ]
    return df.loc[df.index != team].index


def get_wild_card_rivals(standings_immutable, season_params, team):
    standings = standings_immutable.copy()
    # print(f'The standings I got: {standings}')
    my_league = standings.loc[team]["lg"]
    standings = standings.loc[standings["lg"] == my_league]
    standings = standings.loc[~standings["division_leader"]]
    standings = standings.drop(
        standings.nlargest(season_params.wildcard_count - 1, columns=["W"]).index
    )
    return standings.loc[standings.index != team].index


def head_to_head_lookup(head_to_head, team1, team2):
    try:
        if team1 < team2:
            row = head_to_head.loc[team1].loc[team2]
            return row["alpha1_wins"], row["alpha2_wins"]
        else:
            row = head_to_head.loc[team2].loc[team1]
            return row["alpha2_wins"], row["alpha1_wins"]
    except KeyError:
        return 0, 0


def get_elimination_number(
    standings, head_to_head, remaining, season_length, team1, team2
):
    # TODO: Make this support pre-2022 years.
    h2h_wins, h2h_losses = head_to_head_lookup(head_to_head, team1, team2)
    remaining_h2h = remaining["size"].get((min(team1, team2), max(team1, team2)), 0)
    max_h2h_margin = remaining_h2h + h2h_wins - h2h_losses
    if max_h2h_margin == 0:
        if remaining_h2h == 0:
            print(
                f"Um there is a guaranteed head-to-head tie between {team1} and {team2} and I am not sure how to handle that."
            )
        else:
            print(
                f"Um there might be a head-to-head tie between {team1} and {team2} and I am not sure how to handle that."
            )
    okay_to_tie = max_h2h_margin > 0
    if "OAK" in [team1, team2] and "HOU" in [team1, team2]:
        print(
            f"for HOU and OAK, h2h_wins {h2h_wins} h2h_losses {h2h_losses} remaining_h2h {remaining_h2h}"
        )
    our_losses = standings.loc[team1]["L"]
    their_wins = standings.loc[team2]["W"]
    naive_elimination = (season_length - our_losses) - their_wins
    if okay_to_tie:
        return naive_elimination
    else:
        return naive_elimination - 1


def all_elimination_numbers(
    standings, head_to_head, remaining, remaining_indexed, season_length, team, rivals
):
    eliminations = {
        rival: get_elimination_number(
            standings, head_to_head, remaining_indexed, 162, team, rival
        )
        for rival in rivals
    }
    set_eliminations = []
    for subset in powerset(rivals):
        if len(subset) < 1:
            continue
        subset_matchup_count = remaining.loc[
            (remaining["alpha1"].isin(subset)) & (remaining["alpha2"].isin(subset))
        ]["size"].sum()
        if len(subset) > 1 and (not subset_matchup_count):
            continue
        total_elimination = sum([eliminations[rival] for rival in subset])
        new_tuple = (
            total_elimination - subset_matchup_count,
            team,
            subset,
            total_elimination,
            subset_matchup_count,
        )
        if new_tuple[0] < 20:
            set_eliminations.append(new_tuple)
    return set_eliminations


def get_division_contenders3(
    played_orig, unplayed, season_params: SeasonParameters, date=None
):
    if not date:
        date = played_orig["completion_date"].max()
    played = played_orig.loc[played_orig["completion_date"] <= date]

    standings = compute_standings(played)
    head_to_head = head_to_head_records(played)
    remaining = (
        all_matchups_after_date(played_orig, unplayed, date)
        .groupby(["alpha1", "alpha2"], as_index=False)
        .size()
    )
    remaining_indexed = remaining.set_index(["alpha1", "alpha2"])
    set_eliminations = []
    for team in standings.index:
        rivals = get_division_rivals(standings, season_params.divisions, team)
        set_eliminations += all_elimination_numbers(
            standings, head_to_head, remaining, remaining_indexed, 162, team, rivals
        )
    standings_wc = wildcard_standings(standings, season_params)
    for team in standings.index:
        rivals = get_wild_card_rivals(standings_wc, season_params, team)
        set_eliminations += all_elimination_numbers(
            standings, head_to_head, remaining, remaining_indexed, 162, team, rivals
        )
    return sorted(set_eliminations)


def run_elo():
    ELO = pd.read_csv("./data/mlb_elo_latest.csv")
    PLAYED, UNPLAYED = elo_to_played_and_unplayed(ELO)
    NICKNAMES = load_nicknames("data/CurrentNames.csv")
    TEAM_IDS = load_team_ids("data/TEAMABR.TXT")
    SP2022 = SeasonParameters(NICKNAMES, TEAM_IDS, 2022)
    show_dumb_elimination_output4(PLAYED, UNPLAYED, SP2022)


def run_statsapi():
    teams_dict = json.load(open("./data/teams.json"))
    PLAYED, UNPLAYED = statsapi_schedule_to_played_unplayed(
        "./data/schedule_2022_14.json"
    )
    print(
        PLAYED.loc[
            (PLAYED["home"] == "Cleveland Guardians")
            | (PLAYED["visitor"] == "Cleveland Guardians")
        ]
    )
    print(
        UNPLAYED.loc[
            (UNPLAYED["home"] == "Cleveland Guardians")
            | (UNPLAYED["visitor"] == "Cleveland Guardians")
        ]
    )
    SP2022 = SeasonParameters(
        2022,
        statsapi_played=PLAYED,
        statsapi_unplayed=UNPLAYED,
        statsapi_teams=teams_dict,
    )
    show_dumb_elimination_output4(PLAYED, UNPLAYED, SP2022)


def run_statsapi_just_elimination_numbers():
    teams_dict = json.load(open("./data/teams.json"))
    PLAYED, UNPLAYED = statsapi_schedule_to_played_unplayed(
        "./data/schedule_2022_1.json"
    )
    SP2022 = SeasonParameters(
        2022,
        statsapi_played=PLAYED,
        statsapi_unplayed=UNPLAYED,
        statsapi_teams=teams_dict,
    )
    for thingamabob in get_division_contenders3(PLAYED, UNPLAYED, SP2022):
        print(thingamabob)


if __name__ == "__main__":
    # run_one_year_retro(1964)
    # run_elo()
    run_statsapi_just_elimination_numbers()
