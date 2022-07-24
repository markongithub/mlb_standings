#!/usr/bin/env python

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

# TODO: 1899 teams are not in here, I need to use another one.
def divisions_for_year(nicknames_from_retrosheet, team_ids_from_retrosheet, year):
    if year <= 1915:
        return divisions_from_team_ids(team_ids_from_retrosheet, year)
    df = nicknames_from_retrosheet.copy()
    # df['start_date'] = df['start_date'].astype(str)
    df['start_year'] = pd.DatetimeIndex(df['start_date']).year
    df['end_year'] = pd.DatetimeIndex(df['end_date']).year
    df = df.loc[(df['start_year'] <= year) & ((df['end_year'].isna()) | (df['end_year'] >= year))] # & (df['end_year'] => year)]
    df['div'] = df[['league', 'division']].fillna('').sum(axis=1)
    
    return df.set_index('contemporary_id')[['league', 'div']].rename(columns={'league': 'lg'})

def divisions_from_team_ids(team_ids_from_retrosheet, year):
    df = team_ids_from_retrosheet.copy()
    df = df.loc[(df['start_year'] <= year) & (df['end_year'] >= year)]
    df['div'] = df['league'] # the league is the division is the league
    return df.set_index('team_id')[['league', 'div']].rename(columns={'league': 'lg'})

def division_contenders(standings_immutable, divisions, season_length):
    df = standings_immutable.copy()
#    print(f'division_contenders was called with columns {df.columns}')
    df['div'] = divisions['div']
    max_wins = season_length - df['L']
    division_contenders = set()
    for index, row in df.iterrows():
        most_wins_in_division = df.loc[(df['div'] == row["div"])]['W'].max()
        if max_wins[index] >= most_wins_in_division:
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


def divisional_threats(standings_immutable, divisions, season_length, team):
    # We only care about teams with more max_wins than us.
    df = standings_immutable.copy()
    my_division = divisions.loc[team]['div']
    df = df.merge(divisions, left_index=True, right_index=True).loc[divisions['div'] == my_division]   
    max_wins = season_length - df['L']
    df = df.loc[(df.index != team) & (max_wins[df.index] >= max_wins[team])]
    return max_wins[team] - df['W']

def games_between_rivals_after_date(df, schedule, divisions, date_str, team):
    remaining = all_matchups_after_date(df, schedule, date_str).groupby(['alpha1', 'alpha2'], as_index=False).size()
    standings = compute_standings(df.loc[(df['completion_date'] <= date_str)])
    season_length = get_season_length(schedule) # why god why
    threats = divisional_threats(standings, divisions, season_length, team)
    return remaining.loc[(remaining['alpha1'].isin(threats.index)) & (remaining['alpha2'].isin(threats.index))]

def get_season_length(schedule):
    unique_game_counts = pd.unique(pd.concat([schedule['home'], schedule['visitor']], axis = 0).value_counts())
    game_counts = pd.concat([schedule['home'], schedule['visitor']], axis = 0).value_counts()
    assert unique_game_counts.shape == (1,), f'season lengths are all messed up: {game_counts}'
    return unique_game_counts[0]

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

def is_division_contender_with_rivalries(game_log, schedule, divisions, date_str, team):
    matchups = games_between_rivals_after_date(game_log, schedule, divisions, date_str, team)
    season_length = get_season_length(schedule) # why god why
    threats = divisional_threats(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, season_length, team)
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats)

def wildcard_standings(standings_immutable, divisions, wildcard_count):
    contenders = set()
    if not wildcard_count:
        return pd.DataFrame()
    df = standings_immutable.copy()
    # I hate myself for this
    df['div'] = divisions['div']
    df['lg'] = divisions['lg']
    df['max_wins'] = 162 - df['L']

    df['first_place'] = False
    df.loc[df.groupby('div').head(1).index, 'first_place'] = True
    wildcard_wins_by_league = df.loc[df['first_place'] == False].sort_values(by=['W'], ascending=False).groupby('lg').nth(wildcard_count - 1)['W']
    def this_is_horrible(lg):
        return wildcard_wins_by_league.loc[lg]
    merge1 = df.merge(wildcard_wins_by_league, left_on=['lg'], right_index=True)
    return merge1.loc[merge1['max_wins'] >= merge1['W_y']][['W_x', 'L', 'div', 'lg', 'max_wins', 'first_place']].rename(columns={'W_x': 'W'})


def wildcard_contenders_naive(standings_immutable, divisions, wildcard_count):
    return set(wildcard_standings(standings_immutable, divisions, wildcard_count).index)

def wildcard_threats(standings_immutable, divisions, team, wildcard_count):
    # We only care about teams with more max_wins than us
    df = standings_immutable.copy()
    # this is so bad
    df['lg'] = divisions['lg']

    my_league = divisions.loc[team]['lg']
    my_max_wins = df.loc[team]['max_wins']
    df = df.loc[(df['lg'] == my_league) & (df['first_place'] == False)]
    df = df.drop(df.nlargest(wildcard_count - 1, columns=['W']).index)
    return my_max_wins - df['W']

def games_between_wildcard_rivals_after_date(df, schedule, divisions, date_str, team, wildcard_count):
    remaining = all_matchups_after_date(df, schedule, date_str).groupby(['alpha1', 'alpha2'], as_index=False).size()
    standings = wildcard_standings(compute_standings(df.loc[(df['completion_date'] <= date_str)]), divisions, wildcard_count)
    threats = wildcard_threats(standings, divisions, team, wildcard_count)
    return remaining.loc[(remaining['alpha1'].isin(threats.index)) & (remaining['alpha2'].isin(threats.index))]

def is_wildcard_contender_with_rivalries(game_log, schedule, divisions, date_str, team, wildcard_count):
    matchups = games_between_wildcard_rivals_after_date(game_log, schedule, divisions, date_str, team, wildcard_count)
    standings = wildcard_standings(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, wildcard_count)
    # print(f'About to call wildcard_threats where standings has these columns: {standings.columns}')
    threats = wildcard_threats(standings, divisions, team, wildcard_count)
    sorted_rivals = sort_rivals(matchups, threats)
    return all_subset_sums(sorted_rivals, threats)

def count_teams(game_log, schedule, divisions):
    game_log_teams = pd.unique(game_log[['home', 'visitor']].values.ravel('K'))
    schedule_teams = pd.unique(game_log[['home', 'visitor']].values.ravel('K'))
    divisions_count = len(divisions.index)
    if (len(game_log_teams) != len(schedule_teams) or len(game_log_teams) != divisions_count):
        print(f'Teams in game log: {sorted(pd.unique(game_log[["home", "visitor"]].values.ravel("K")))} ')
        print(f'Teams in schedule: {sorted(pd.unique(schedule[["home", "visitor"]].values.ravel("K")))} ')
        print(f'Teams in division map: {sorted(divisions.index)}')
    assert(sorted(game_log_teams) == sorted(schedule_teams))
    assert(all(t in divisions.index for t in game_log_teams))
    return len(game_log_teams)

def show_dumb_elimination_output3(df, schedule, divisions, wildcard_count=2):
    team_count = count_teams(df, schedule, divisions)
    games_per_season = get_season_length(schedule)
    print(f'This season has {team_count} teams and {games_per_season} for each team.')
    # If home_game_num > games_per_season this is probably a tiebreaker playoff.
    max_date = retro_to_datetime(df.loc[df['home_game_num'] <= games_per_season]['completion_date'].max())
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
        current_standings['max_wins'] = games_per_season - current_standings['L']
        # print(current_standings)
    
        div_contenders = division_contenders(current_standings, divisions, games_per_season)
        # print(f'naive division contenders: {div_contenders}')
        for supposed_contender in div_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if tomorrows_div_contenders and supposed_contender not in tomorrows_div_contenders:
                if not is_division_contender_with_rivalries(df, schedule, divisions, date_str, supposed_contender):
                    print(f'It looked like {supposed_contender} was in contention after {date_str} but the remaining intra-division games ruled them out.')
                    div_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_div_contenders:
            new_contenders = div_contenders.difference(tomorrows_div_contenders)
        # if new_contenders:
            # print(f'Teams eliminated from their division titles on {datetime_to_retro(tomorrow)}: {new_contenders}')
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = games_per_season - tomorrows_standings.loc[eliminated_team]['W'] - tomorrows_standings.loc[eliminated_team]['L']
            elim_div = divisions.loc[eliminated_team]['div'] 
            print(f'{eliminated_team} were eliminated from the {elim_div} title on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play.')
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            if eliminations.get(eliminated_team):
                eliminations[eliminated_team]['division'] = new_pair
            else:
                eliminations[eliminated_team] = {'division': new_pair}
        # print(f'PHI max wins: {current_standings.loc["PHI"]["max_wins"]}, SLN wins: {current_standings.loc["SLN"]["W"]}')
        # print(f'My busted view of the wildcard standings: {wildcard_standings(current_standings)}')
        wildcard_contenders = wildcard_contenders_naive(current_standings, divisions, wildcard_count)
        # print(f'naive wildcard contenders after {date_str} games: {wildcard_contenders}')
        for supposed_contender in wildcard_contenders.copy():
            # If you're in contention tomorrow, you're in contention today, so I am not going to
            # waste CPU time on you.
            if tomorrows_wildcard_contenders and supposed_contender not in tomorrows_wildcard_contenders:
                # print(f'let\'s see if {supposed_contender} is really still in wildcard contention')
                if not is_wildcard_contender_with_rivalries(df, schedule, divisions, date_str, supposed_contender, wildcard_count):
                    print(f'It looked like {supposed_contender} was in wildcard contention after {date_str} but the remaining intra-contender games ruled them out.')
                    wildcard_contenders.remove(supposed_contender)
        new_contenders = set()
        if tomorrows_wildcard_contenders:
            new_contenders = wildcard_contenders.difference(tomorrows_wildcard_contenders)
        for eliminated_team in new_contenders:
            games_to_go_at_elimination = games_per_season - tomorrows_standings.loc[eliminated_team]['W'] - tomorrows_standings.loc[eliminated_team]['L']
            elim_lg = divisions.loc[eliminated_team]['lg'] 

            print(f'{eliminated_team} were eliminated from {elim_lg} wildcard contention on {datetime_to_retro(tomorrow)} with {games_to_go_at_elimination} games left to play.')
            new_pair = (datetime_to_retro(tomorrow), games_to_go_at_elimination)
            if eliminations.get(eliminated_team):
                eliminations[eliminated_team]['wildcard'] = new_pair
            else:
                eliminations[eliminated_team] = {'wildcard': new_pair}


        contenders_any = div_contenders.union(wildcard_contenders)
        if tomorrows_contenders_any:
            new_contenders = contenders_any.difference(tomorrows_contenders_any)
        if new_contenders:
            print(f'Teams eliminated from ALL postseason contention on {datetime_to_retro(tomorrow)}: {new_contenders}')
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
    if year == 2020:
        return "2020 was weird sorry"
    if year >= 2012:
        return 2
    if year >= 1994:
        return 1
    if year == "1994":
        return "1994 was weird sorry"
    return 0

def run_one_year(year):
    nicknames = load_nicknames('data/CurrentNames.csv')
    team_ids = load_team_ids('data/TEAMABR.TXT')
    return show_dumb_elimination_output3(load_game_log(f'./data/GL{year}.TXT'), load_schedule(f'./data/{year}SKED.TXT'), divisions_for_year(nicknames, team_ids, year), wildcard_count=wildcards_for_year(year))



# In[50]:


# test case - the Mets after 1964-08-14
# and every date up to August 28 - they are obviously out on the 29th
print('can I execute anything at the top of this cell?')
mydict = {}
if mydict:
    print('an empty dict is true')
else:
    print('an empty dict is false')
    
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
    season_length = get_season_length(schedule)
    matchups = games_between_rivals_after_date(game_log, schedule, divisions, date_str, team)
    threats = divisional_threats(compute_standings(game_log.loc[(game_log['completion_date'] <= date_str)]), divisions, season_length, team)
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
    easy_mode = is_division_contender_with_rivalries(game_log, schedule, divisions, date_str, team)
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
bad_years2 [1882, 1884, 1886, 1887, 1961]

