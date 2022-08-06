#!/usr/bin/env python

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_game_log(game_log_path):
    retro_df_columns=['date', 'doubleheader_index','weekday', 'visitor', 'visitor_league', 'visitor_game_num', 'home', 'home_league', 'home_game_num', 'visitor_score', 'home_score', 'length_outs', 'day_night', 'completion', 'forfeit']
    df = pd.read_csv(game_log_path, header=None, names=retro_df_columns, usecols=range(len(retro_df_columns)))
    df['date'] = df['date'].astype(str)
    df = df.fillna(value={'completion': 'I hate Pandas'})
    df['completion'] = df['completion'].astype(str)
    df['completion_date'] = df['date']
    df.loc[(df['completion'] != "I hate Pandas"), 'completion_date'] = df['completion'].str.split(',').str[0]
    df['completion_date'] = df['completion_date'].astype(str)
    return df

def fix_1880(schedule):
    # Baseball Reference calls them the Reds so I am going to coerce this code
    # to call them the Reds.
    schedule.loc[schedule['visitor'] == 'CN1', 'visitor'] = 'CN4'
    schedule.loc[schedule['home'] == 'CN1', 'home'] = 'CN4'
    return schedule

def fix_1882(game_log):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log['visitor'] == 'BL5', 'visitor'] = 'BL2'
    game_log.loc[game_log['home'] == 'BL5', 'home'] = 'BL2'
    return game_log

def fix_1884(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log = game_log.loc[(game_log['visitor_league'] != 'UA') & (game_log['home_league'] != 'UA')]
    game_log.loc[game_log['visitor'] == 'WS7', 'visitor'] = 'RIC'
    game_log.loc[game_log['home'] == 'WS7', 'home'] = 'RIC'
    schedule = schedule.loc[(schedule['visitor_league'] != 'UA') & (schedule['home_league'] != 'UA')]
    schedule.loc[schedule['visitor'] == 'WS7', 'visitor'] = 'RIC'
    schedule.loc[schedule['home'] == 'WS7', 'home'] = 'RIC'
    return game_log, schedule

def fix_1890(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log['visitor'] == 'BR4', 'visitor'] = 'BL3'
    game_log.loc[game_log['home'] == 'BR4', 'home'] = 'BL3'
    schedule.loc[schedule['visitor'] == 'BR4', 'visitor'] = 'BL3'
    schedule.loc[schedule['home'] == 'BR4', 'home'] = 'BL3'
    return game_log, schedule

def fix_1891(game_log, schedule):
    # I guess I'm mutating it in place even though I hate that.
    game_log.loc[game_log['visitor'] == 'CN3', 'visitor'] = 'ML3'
    game_log.loc[game_log['home'] == 'CN3', 'home'] = 'ML3'
    schedule.loc[schedule['visitor'] == 'CN3', 'visitor'] = 'ML3'
    schedule.loc[schedule['home'] == 'CN3', 'home'] = 'ML3'
    return game_log, schedule

# adapted from sdvinay except he didn't think of ties or forfeits
def compute_standings(game_log):
    gms_played = game_log.copy()
    # Eliminate ties but not forfeits.
    gms_played = gms_played[(gms_played['visitor_score'] != gms_played['home_score']) | (gms_played['forfeit'].notna())]
    gms_played = gms_played[gms_played['forfeit'] != "T"]
    margins = gms_played['visitor_score']-gms_played['home_score']
    forfeit = gms_played['forfeit']
    winners = pd.Series(np.where((gms_played['forfeit'] != 'H') & (margins>0) | (gms_played['forfeit'] == 'V'), gms_played['visitor'], gms_played['home']))
    losers  = pd.Series(np.where((gms_played['forfeit'] != 'V') & (margins<0) | (gms_played['forfeit'] == 'H'), gms_played['visitor'], gms_played['home']))
    standings = pd.concat([winners.value_counts().rename('W'), losers.value_counts().rename('L')], axis=1)
    return standings.fillna(0)

# get this from https://www.retrosheet.org/Nickname.htm
# It doesn't have a lot of early stuff.
def load_nicknames(nicknames_file):
    retro_nickname_columns=['current_id', 'contemporary_id', 'league', 'division', 'location', 'nickname', 'alt_nickname', 'start_date', 'end_date']
    df = pd.read_csv(nicknames_file, header=None, names=retro_nickname_columns, usecols=range(len(retro_nickname_columns)))
    return df

def load_team_ids(team_ids_file):
    retro_team_id_columns=['team_id', 'league', 'location', 'nickname', 'start_year', 'end_year']
    df = pd.read_csv(team_ids_file, header=None, names=retro_team_id_columns, usecols=range(len(retro_team_id_columns)))
    return df

def divisions_for_year(nicknames_from_retrosheet, team_ids_from_retrosheet, year):
    if year <= 1915:
        return divisions_from_team_ids(team_ids_from_retrosheet, year)
    df = nicknames_from_retrosheet.copy()
    # df['start_date'] = df['start_date'].astype(str)
    df['start_year'] = pd.DatetimeIndex(df['start_date']).year
    df['end_year'] = pd.DatetimeIndex(df['end_date']).year
    df = df.loc[(df['start_year'] <= year) & ((df['end_year'].isna()) | (df['end_year'] >= year))] # & (df['end_year'] => year)]
    df['div'] = df[['league', 'division']].fillna('').sum(axis=1)

    return df.set_index('contemporary_id')[['league', 'div', 'current_id', 'location', 'nickname']].rename(columns={'league': 'lg', 'current_id': 'franchise_id'})

def divisions_from_team_ids(team_ids_from_retrosheet, year):
    df = team_ids_from_retrosheet.copy()
    df = df.loc[(df['start_year'] <= year) & (df['end_year'] >= year)]
    df['div'] = df['league'] # the league is the division is the league
    df['franchise_id'] = df['team_id']
    return df.set_index('team_id')[['league', 'div', 'franchise_id', 'location', 'nickname']].rename(columns={'league': 'lg'})

def division_contenders(standings_immutable, divisions, season_length, division_winners=1):
    df = standings_immutable.copy()
#    print(f'division_contenders was called with columns {df.columns}')
    #df['div'] = divisions['div']
    max_wins = df['season_length'] - df['L']
    division_contenders = set()
    for index, row in df.iterrows():
        division_win_threshold = df.loc[(df['div'] == row["div"])]['W'].nlargest(division_winners).min()
        # print(f'{index} will need {division_win_threshold} wins to win this division.')
        if max_wins[index] >= division_win_threshold:
            division_contenders.add(index)
    return division_contenders

def retro_to_datetime(retro_str):
    return datetime.strptime(retro_str, '%Y%m%d')

def datetime_to_retro(dt):
    return datetime.strftime(dt, '%Y%m%d')

def load_schedule(filename):
    schedule_columns=['date', 'doubleheader_index','weekday', 'visitor', 'visitor_league', 'visitor_game_num', 'home', 'home_league', 'home_game_num', 'day_night', 'completion', 'makeup_date']
    df = pd.read_csv(filename, header=None, names=schedule_columns, usecols=range(len(schedule_columns)))
    df['completion'] = df['completion'].astype(str)
    df['makeup_date'] = df['makeup_date'].astype(str)
    return df.dropna(subset=['home'])

schedule_path = './data/2021SKED.TXT'
SCHEDULE = load_schedule(schedule_path)

def find_unplayed_games(schedule):
    return schedule.loc[(schedule["makeup_date"].str.startswith("not", na=False)) | (schedule["completion"].str.contains("No makeup", na=False))]

def logged_games_after_date(df, date_str):
    return df.loc[df['completion_date'] > date_str]

def all_matchups_after_date(df, schedule, date_str):
    logged_games = logged_games_after_date(df, date_str)[['date', 'doubleheader_index', 'visitor','home']]
    unplayed_games = find_unplayed_games(schedule)[['date', 'doubleheader_index', 'visitor','home']]
    all_games = pd.concat([logged_games, unplayed_games], ignore_index=True)
    alpha_pairs = pd.DataFrame(np.where(all_games['visitor'] < all_games['home'],
                                     (all_games['visitor'], all_games['home']),
                                     (all_games['home'], all_games['visitor'])))
    return alpha_pairs.T.rename(columns={0: 'alpha1', 1: 'alpha2'})


def divisional_threats(standings_immutable, divisions, season_length, winners_per_division, team):
    # We only care about teams with more max_wins than us.
    df = standings_immutable.copy()
    my_division = divisions.loc[team]['div']
    df = df.merge(divisions, left_index=True, right_index=True).loc[divisions['div'] == my_division]
    max_wins = season_length - df['L']
    df['max_wins'] = max_wins # please kill me
    df = df.loc[(df.index != team) & (max_wins[df.index] >= max_wins[team])]
    # So now we have all the teams with more max wins than us.
    # We don't care about the top n-1 where winners_per_division is n.
    # I don't think it's sorted yet.
    df = df.sort_values(by=['max_wins'], ascending=False).iloc[(winners_per_division-1):]
    return max_wins[team] - df['W']

def games_between_rivals_after_date(df, schedule, divisions, winners_per_division, date_str, team):
    remaining = all_matchups_after_date(df, schedule, date_str).groupby(['alpha1', 'alpha2'], as_index=False).size()
    standings = compute_standings(df.loc[(df['completion_date'] <= date_str)])
    season_lengths = get_season_length(schedule) # why god why
    threats = divisional_threats(standings, divisions, season_lengths, winners_per_division, team)
    return remaining.loc[(remaining['alpha1'].isin(threats.index)) & (remaining['alpha2'].isin(threats.index))]

# I thought there was one season length per season but it's really per league
# per season.
def get_season_length(schedule):
    home = schedule[['home', 'home_league']].rename(columns={'home': 'team', 'home_league' : 'league'})
    visitors = schedule[['visitor', 'visitor_league']].rename(columns={'visitor': 'team', 'visitor_league' : 'league'})
    # if pd.concat([home, visitors]).value_counts().groupby("league").nunique().eq(1)
    grouped = pd.concat([home, visitors]).value_counts().groupby("league")
    if grouped.nunique().eq(1).all():
        return grouped.min()
    else:
        assert(grouped.min() == grouped.max())

# sort the output of division_threats
def sort_rivals(matchups, threats):
    # print(matchups.columns, threats.shape)
    df = matchups.copy()
    merge1 = df.merge(threats, left_on=['alpha1'], right_index=True)
    # print(f'now merge1 is: {merge1}')
    merge2 = merge1.merge(threats, left_on=['alpha2'], right_index=True)
    # print(f'now merge2 is: {merge2}')
    merge2['betterT'] = np.where(merge2['W_x'] >= merge2['W_y'], merge2['alpha2'], merge2['alpha1'])
    merge2['worseT'] = np.where(merge2['W_x'] >= merge2['W_y'], merge2['alpha1'], merge2['alpha2'])
    merge2['lesserW'] = merge2[['W_x','W_y']].min(axis=1)
    merge2['greaterW'] = merge2[['W_x','W_y']].max(axis=1)
    return merge2.sort_values(['lesserW', 'greaterW'], ascending=True)[['betterT', 'worseT', 'size']]

# This never worked reliably. I'm leaving it here for historical reasons in
# case someone ever wants to revisit this approach.
def portion_out_wins(sorted_rivals, threats):
    for _, row in sorted_rivals.iterrows():
        games = row['size']
        betterT = row['betterT']
        worseT = row['worseT']
        threat1, threat2 = threats[betterT], threats[worseT]
        print(f'We can afford to let {betterT} win {threat1} more games, and {worseT} {threat2}')
        if games > threat1 + threat2:
            # No matter what, these games will lead to too many wins for one team.
            print(f'But there are {games} games left between {betterT} and {worseT}')
            return False
        # We try to give as many wins to the best team as possible first. Then distribute them
        # downward. Not sure about this.
        # In the Baltimore example, Tampa Bay cannot win another game. So wins_for_t1 is going to be
        # zero. We give all three of the TB-NY series to NY.
        wins_for_t1 = min(threat1, games)
        wins_for_t2 = games - wins_for_t1
        print(f'Let\'s say that {betterT} wins {wins_for_t1} of their remaining {games} with {worseT}')
        threats[row['betterT']] -= wins_for_t1
        threats[row['worseT']] -= wins_for_t2
    return True

def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def all_subset_sums(matchups, threats):
    all_rivals = threats.index
    for subset in powerset(all_rivals):
        if len(subset) < 2:
            continue
        if 'betterT' in matchups.columns:
            team1, team2 = 'betterT', 'worseT'
        else:
            team1, team2 = 'alpha1', 'alpha2'
        subset_matchup_count = matchups.loc[(matchups[team1].isin(subset)) & (matchups[team2].isin(subset))]['size'].sum()
        allowable_win_count = threats.loc[threats.index.isin(subset)].sum()
        if subset_matchup_count <= allowable_win_count:
            # print(f'We can allow {subset} to win {allowable_win_count} and they only have {subset_matchup_count} games left so that seems okay.')
            pass
        else:
            # print(f'We can only allow {subset} to win {allowable_win_count} but they have {subset_matchup_count} games left.')
            return False
    return True

def is_division_contender_with_rivalries(game_log, schedule, divisions, winners_per_division, date_str, team):
    matchups = games_between_rivals_after_date(game_log, schedule, divisions, winners_per_division, date_str, team)
    season_length = get_season_length(schedule).loc[divisions.loc[team]['lg']] # why god why
    threats = divisional_threats(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, season_length, winners_per_division, team)
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats)

def wildcard_standings(standings_immutable, divisions, wildcard_count, division_winners, games_per_season):
    contenders = set()
    if not wildcard_count:
        return pd.DataFrame()
    df = standings_immutable.copy()
    # I hate myself for this
    df['div'] = divisions['div']
    df['lg'] = divisions['lg']
    # I have no idea what I am doing.
    def why_is_this_necessary(league):
        return games_per_season[league]
    df['season_length'] = df['lg'].apply(why_is_this_necessary)

    df['max_wins'] = df['season_length'] - df['L']


    df['division_leader'] = False
    df = df.sort_values(by=['max_wins'], ascending=False)
    df.loc[df.groupby('div').head(division_winners).index, 'division_leader'] = True
    wildcard_wins_by_league = df.loc[df['division_leader'] == False].sort_values(by=['W'], ascending=False).groupby('lg').nth(wildcard_count - 1)['W']
    def this_is_horrible(lg):
        return wildcard_wins_by_league.loc[lg]
    merge1 = df.merge(wildcard_wins_by_league, left_on=['lg'], right_index=True)
    return merge1.loc[merge1['max_wins'] >= merge1['W_y']][['W_x', 'L', 'div', 'lg', 'max_wins', 'division_leader']].rename(columns={'W_x': 'W'})


def wildcard_contenders_naive(standings_immutable, divisions, wildcard_count, winners_per_division, games_per_season):
    return set(wildcard_standings(standings_immutable, divisions, wildcard_count, winners_per_division, games_per_season).index)

def wildcard_threats(standings_immutable, divisions, team, wildcard_count):
    # We only care about teams with more max_wins than us
    df = standings_immutable.copy()
    # this is so bad
    df['lg'] = divisions['lg']

    my_league = divisions.loc[team]['lg']
    my_max_wins = df.loc[team]['max_wins']
    df = df.loc[(df['lg'] == my_league) & (df['division_leader'] == False)]
    df = df.drop(df.nlargest(wildcard_count - 1, columns=['W']).index)
    return my_max_wins - df['W']

def games_between_wildcard_rivals_after_date(df, schedule, divisions, date_str, team, wildcard_count, winners_per_division):
    remaining = all_matchups_after_date(df, schedule, date_str).groupby(['alpha1', 'alpha2'], as_index=False).size()
    basic_standings = compute_standings(df.loc[(df['completion_date'] <= date_str)])
    games_per_season = get_season_length(schedule)
    standings = wildcard_standings(compute_standings(df.loc[(df['completion_date'] <= date_str)]), divisions, wildcard_count, division_winners=winners_per_division, games_per_season=games_per_season)
    threats = wildcard_threats(standings, divisions, team, wildcard_count)
    return remaining.loc[(remaining['alpha1'].isin(threats.index)) & (remaining['alpha2'].isin(threats.index))]

def is_wildcard_contender_with_rivalries(game_log, schedule, divisions, date_str, team, wildcard_count, winners_per_division):
    matchups = games_between_wildcard_rivals_after_date(game_log, schedule, divisions, date_str, team, wildcard_count, winners_per_division)
    games_per_season = get_season_length(schedule)
    standings = wildcard_standings(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, wildcard_count, winners_per_division, games_per_season)
    # print(f'About to call wildcard_threats where standings has these columns: {standings.columns}')
    threats = wildcard_threats(standings, divisions, team, wildcard_count)
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats)

def count_teams(game_log, schedule, divisions):
    game_log_teams = sorted(pd.unique(game_log[['home', 'visitor']].values.ravel('K')))
    schedule_teams = sorted(pd.unique(schedule[['home', 'visitor']].values.ravel('K')))
    divisions_count = len(divisions.index)
    if (game_log_teams != schedule_teams or any(t not in divisions.index for t in game_log_teams)):
        print(f'Teams in game log: {game_log_teams} ')
        print(f'Teams in schedule: {schedule_teams} ')
        print(f'Teams in division map: {sorted(divisions.index)}')
    assert(game_log_teams == schedule_teams)
    assert(all(t in divisions.index for t in game_log_teams))
    return len(game_log_teams)

def display_name(divisions, team_id):
    team_entry = divisions.loc[team_id]
    return f'{team_entry["location"]} {team_entry["nickname"]}'

def show_dumb_elimination_output3(df, schedule, divisions, wildcard_count=2, winners_per_division=1):
    team_count = count_teams(df, schedule, divisions)
    games_per_season = get_season_length(schedule)
    print(f'This season has {team_count} teams.')
    print(f'The top {winners_per_division} teams from each division go to the postseason.')
    for index, value in games_per_season.iteritems():
        print(f'The {index} has {value} games per team.')
    # If home_game_num > games_per_season this is probably a tiebreaker playoff.
    # I guess each league could have their tiebreakers on different days? I hope
    # this doesn't exclude a regular season game. Maybe I should make the check
    # in compute_standings.
    max_date = retro_to_datetime(df.loc[df['home_game_num'] <= games_per_season.max()]['completion_date'].max())
    print(f'max_date is {max_date}')
    min_date = retro_to_datetime(df['completion_date'].min())

    current_date = max_date
    div_contenders = set()
    tomorrows_div_contenders = None
    wildcard_contenders = set()
    tomorrows_wildcard_contenders = None
    tomorrows_contenders_any = None
    tomorrows_standings = None
    eliminations = {}
    while (
        current_date > min_date and        
        ((len(div_contenders) < team_count) or
         wildcard_count and len(wildcard_contenders) < team_count)):
        date_str = datetime_to_retro(current_date)
        # print(f'Starting analysis of {date_str}')
        current_standings = compute_standings(df.loc[df['completion_date'] <= date_str])
        # print(current_standings)
        # MOVE THIS SHIT INTO SOMETHING EFFICIENT
        current_standings['div'] = divisions['div']
        current_standings['lg'] = divisions['lg']
        # I have no idea what I am doing.
        def why_is_this_necessary(league):
            return games_per_season[league]
        current_standings['season_length'] = current_standings['lg'].apply(why_is_this_necessary)
        current_standings['max_wins'] = current_standings['season_length'] - current_standings['L']
        # print(current_standings)
    
        div_contenders = division_contenders(current_standings, divisions, games_per_season, division_winners=winners_per_division)
        # print(f'naive division contenders: {div_contenders}')
        for supposed_contender in div_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if tomorrows_div_contenders and supposed_contender not in tomorrows_div_contenders:
                if not is_division_contender_with_rivalries(df, schedule, divisions, winners_per_division=winners_per_division, date_str=date_str, team=supposed_contender):
                    print(f'It looked like the {display_name(divisions, supposed_contender)} were in contention after {date_str} but the remaining intra-division games ruled them out.')
                    div_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_div_contenders:
            new_contenders = div_contenders.difference(tomorrows_div_contenders)
        # if new_contenders:
            # print(f'Teams eliminated from their division titles on {datetime_to_retro(tomorrow)}: {new_contenders}')
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = tomorrows_standings.loc[eliminated_team]['season_length'] - tomorrows_standings.loc[eliminated_team]['W'] - tomorrows_standings.loc[eliminated_team]['L']
            elim_div = divisions.loc[eliminated_team]['div'] 
            print(f'The {display_name(divisions, eliminated_team)} were eliminated from the {elim_div} title on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play.')
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = divisions.loc[eliminated_team]['franchise_id'] 
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]['division'] = new_pair
            else:
                eliminations[elim_franchise] = {'division': new_pair}
        # print(f'PHI max wins: {current_standings.loc["PHI"]["max_wins"]}, SLN wins: {current_standings.loc["SLN"]["W"]}')
        # print(f'My busted view of the wildcard standings: {wildcard_standings(current_standings, divisions, wildcard_count, division_winners=winners_per_division, games_per_season=games_per_season)}')
        wildcard_contenders = wildcard_contenders_naive(current_standings, divisions, wildcard_count, winners_per_division, games_per_season)
        # print(f'naive wildcard contenders after {date_str} games: {wildcard_contenders}')
        for supposed_contender in wildcard_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if tomorrows_wildcard_contenders and supposed_contender not in tomorrows_wildcard_contenders:
                # print(f'let\'s see if {supposed_contender} is really still in wildcard contention')
                if not is_wildcard_contender_with_rivalries(df, schedule, divisions, date_str, supposed_contender, wildcard_count, winners_per_division):
                    print(f'It looked like the {display_name(divisions, supposed_contender)} were in wildcard contention after {date_str} but the remaining intra-contender games ruled them out.')
                    wildcard_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_wildcard_contenders:
            new_contenders = wildcard_contenders.difference(tomorrows_wildcard_contenders)
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = tomorrows_standings.loc[eliminated_team]['season_length'] - tomorrows_standings.loc[eliminated_team]['W'] - tomorrows_standings.loc[eliminated_team]['L']
            elim_lg = divisions.loc[eliminated_team]['lg'] 

            print(f'The {display_name(divisions, eliminated_team)} were eliminated from {elim_lg} wildcard contention on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play.')
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            elim_franchise = divisions.loc[eliminated_team]['franchise_id'] 
            if eliminations.get(elim_franchise):
                eliminations[elim_franchise]['wildcard'] = new_pair
            else:
                eliminations[elim_franchise] = {'wildcard': new_pair}


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



# In[47]:


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

def run_one_year(year):
    print(f'starting analysis of {year}')
    nicknames = load_nicknames('data/CurrentNames.csv')
    team_ids = load_team_ids('data/TEAMABR.TXT')
    game_log = load_game_log(f'./data/GL{year}.TXT')
    if year == 2020:
        schedule = load_schedule('./data/2020REV.TXT')
    else:
        schedule = load_schedule(f'./data/{year}SKED.TXT')
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
    return show_dumb_elimination_output3(game_log, schedule, divisions_for_year(nicknames, team_ids, year), wildcard_count=wildcards_for_year(year), winners_per_division=get_winners_per_division(year))



# In[50]:


# test case - the Mets after 1964-08-14
# and every date up to August 28 - they are obviously out on the 29th
    
def assign_wins_with_brute_force(sorted_rivals, threats_immutable, recursion_level=0):
    def myprint(mytext):
        if recursion_level <= 15:
            print (' ' * 2 * recursion_level, mytext)
    threats = threats_immutable.copy()
    if sorted_rivals.empty:
        return {}
    if all_subset_sums(sorted_rivals.rename(columns={'betterT': 'alpha1', 'worseT': 'alpha2'}), threats):
        pass
    else:
        return None

    row = sorted_rivals.iloc[0]
    remaining_rows = sorted_rivals.iloc[1: , :]
    games = row['size']
    betterT = row['betterT']
    worseT = row['worseT']
    threat1, threat2 = threats[betterT], threats[worseT]
    myprint(f'We can afford to let {betterT} win {threat1} more games, and {worseT} {threat2}')
    if threat1 < 0 or threat2 < 0:
        myprint(f'How the fuck did that get negative?')
        return None
    if games > threat1 + threat2:
        # No matter what, these games will lead to too many wins for one team.
        myprint(f'But there are {games} games left between {betterT} and {worseT}')
        return None
    minWinsForT1 = max(0, games - threat2)
    maxWinsForT1 = min(threat1, games)
    myprint(f'{betterT} has to win between {minWinsForT1} and {maxWinsForT1} of these {games} games.')
    for winsForT1 in range(minWinsForT1, maxWinsForT1+1):
        winsForT2 = games - winsForT1
        myprint(f'What if {betterT} won {winsForT1} out of {games} and {worseT} won {winsForT2}?')
        if winsForT1 > threat1:
            myprint(f'That is too many wins for {betterT} so we\'re done here.')
            return None
        if winsForT2 > threat2:
            myprint(f'That is too many wins for {worseT}.')
            continue
        # get ready to recurse
        new_threats = threats.copy()
        new_threats[betterT] -= winsForT1
        new_threats[worseT] -= winsForT2
        recurse = assign_wins_with_brute_force(remaining_rows, new_threats, recursion_level+1)
        if recurse != None:
            recurse[(betterT, worseT)] = winsForT1
            return recurse
        # If that recursive check was false, we'll keep going on the for loop.
    return None

def check_division_contention(date_str, year, team):
    game_log = load_game_log(f'./data/GL{year}.TXT')
    schedule = load_schedule(f'./data/{year}SKED.TXT')
    divisions = divisions_for_year(NICKNAMES, TEAM_IDS_UNDEFINED_LOL, year)
    season_lengths = get_season_length(schedule)
    matchups = games_between_rivals_after_date(game_log, schedule, divisions, date_str, team)
    threats = divisional_threats(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, season_lengths, team)
    sum_mode = dumb_matrix_sum(matchups, threats)
    print(f'Sum mode says {sum_mode}')
    # sum mode errs on the side of True. If it's False, brute force will never return True.
    if not sum_mode:
        return False
    subset_mode = all_subset_sums(matchups, threats)
    print(f'Subset mode says {subset_mode}')
    # subset mode errs on the side of True. If it's False, brute force will never return True.
    if not subset_mode:
        return False
    sorted_rivals = sort_rivals(matchups, threats)
    easy_mode = is_division_contender_with_rivalries(game_log, schedule, divisions, winners_per_division, date_str, team)
    # print(f'Easy mode says {easy_mode}')
    # Easy mode errs on the side of False. If it's true, brute force will never return False.
    # if easy_mode:
    #    return True
    brute_force = brute_force_rival_matchups(sorted_rivals, threats)
    print('brute force completed')
    return brute_force

def dumb_matrix_sum(matchups, threats):
    total_matchups = matchups['size'].sum()
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

#print(run_one_year(2020))
#for year in [2021]:
#    output = run_one_year(year)
#    with open(f'output/{year}.json', 'w') as fp:
#      json.dump(output, fp, cls=NpEncoder)

