from flask import make_response
import time
import pandas as pd
import requests
import datetime
import os
import sys
import traceback
import numpy as np
import json

# Determine the correct data path based on environment
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Railway sets RAILWAY_ENVIRONMENT variable when deployed
if os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY'):
    # Running on Railway - use data directory in project root
    DATA_PATH = os.path.join(script_dir, 'data')
elif os.environ.get('VERCEL_ENV'):
    # Running on Vercel - try different possible paths
    if os.path.exists(os.path.join(script_dir, '../data')):
        DATA_PATH = os.path.join(script_dir, '../data')
    elif os.path.exists('/var/task/data'):
        DATA_PATH = '/var/task/data'
    elif os.path.exists('data'):
        DATA_PATH = 'data'
    else:
        DATA_PATH = 'data'
        print(f"Warning: Could not find data directory on Vercel")
else:
    # Running locally - use absolute path relative to script
    DATA_PATH = os.path.join(script_dir, 'data')

print(f"\nUsing DATA_PATH: {DATA_PATH}")
print(f"DATA_PATH exists: {os.path.exists(DATA_PATH)}")

# Verify the path exists and has CSV files
if os.path.exists(DATA_PATH):
    csv_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {DATA_PATH}")

# Load data that's needed by multiple functions
try:
    teams = pd.read_csv(f'{DATA_PATH}/Team_Projected_Data.csv')
    intercepts = pd.read_csv(f'{DATA_PATH}/Projected_Intercepts.csv')
except FileNotFoundError as e:
    error_response = {
        'error': str(e),
        'data_path': DATA_PATH,
        'data_path_exists': os.path.exists(DATA_PATH)
    }
    print(f"Error loading data files: {error_response}")

home_ff_int = intercepts['home_ff_int'].iloc[0]
home_gf_int = intercepts['home_gf_int'].iloc[0]
away_ff_int = intercepts['away_ff_int'].iloc[0]
away_gf_int = intercepts['away_gf_int'].iloc[0]

def compute_home_goal_away_goal_probs(home_team, away_team):
    home_team = np.where(home_team=='TBL', 'T.B', 
    (np.where(home_team=='SJS', 'S.J',
    (np.where(home_team=='LAK', 'L.A',
    (np.where(home_team=='NJD', 'N.J',
    home_team))))))).item()

    away_team = np.where(away_team=='TBL', 'T.B', 
        (np.where(away_team=='SJS', 'S.J',
        (np.where(away_team=='LAK', 'L.A',
        (np.where(away_team=='NJD', 'N.J',
        away_team))))))).item()

    away = teams[teams.team==away_team].reset_index(drop = True)
    away.columns = 'away_' + away.columns
    
    home = teams[teams.team==home_team].reset_index(drop = True)
    home.columns = 'home_' + home.columns

    schedule = pd.concat([away, home], axis = 1)

    schedule = schedule.assign(home_ff = home_ff_int + schedule.home_projected_ff + schedule.away_projected_fa,
               home_xgf = home_gf_int + schedule.home_projected_xgf + schedule.away_projected_xga,
               away_ff = away_ff_int + schedule.away_projected_ff + schedule.home_projected_fa, 
               away_xgf = away_gf_int + schedule.away_projected_xgf + schedule.home_projected_xga,
               home_shooting = schedule.home_projected_shooting + schedule.away_projected_saving,
               away_shooting = schedule.away_projected_shooting + schedule.home_projected_saving)
    
    schedule = schedule.assign(home_sh_perc = schedule.home_shooting + 
                               #xgf_int/ff_int,
                               (schedule.home_xgf/schedule.home_ff),
                   away_sh_perc = schedule.away_shooting + 
                               #xgf_int/ff_int)
                                + (schedule.away_xgf/schedule.away_ff))
    
    schedule = schedule.assign(home_goal_prob = ((schedule.home_ff/3600) * schedule.home_sh_perc),
        away_goal_prob = ((schedule.away_ff/3600) * schedule.away_sh_perc))
    
    home_goal_prob = schedule.home_goal_prob.iloc[0]
    away_goal_prob = schedule.away_goal_prob.iloc[0]

    return home_goal_prob, away_goal_prob

def simulate_regular_season_ot(home_goal_prob, away_goal_prob, seconds_remaining_in_ot = 300):

    while True: # Just in case they score at same second of OT
        
        seconds_until_home_scores = np.random.geometric(home_goal_prob)
        seconds_until_away_scores = np.random.geometric(away_goal_prob)
        if seconds_until_home_scores < seconds_until_away_scores and seconds_until_home_scores <= seconds_remaining_in_ot:
            return 'Home Wins in OT'
        if seconds_until_home_scores > seconds_until_away_scores and seconds_until_away_scores <= seconds_remaining_in_ot:
            return 'Away Wins in OT'
        # Simulate shootout if both teams didnt score in OT
        if seconds_until_home_scores > seconds_remaining_in_ot and seconds_until_away_scores > seconds_remaining_in_ot:
            res = np.random.binomial(n = 1, p = 1/2, size = 1)[0]
            if res == 1:
                return 'Home Wins in Shootout'
            else:
                return 'Away Wins in Shootout'

def simulate_live_game_in_regulation(home_team, away_team, simulations, initial_home_goals, initial_away_goals, seconds_remaining_in_regulation = 3600):

    home_goal_prob, away_goal_prob = compute_home_goal_away_goal_probs(home_team, away_team)

    home_goals = np.random.binomial(seconds_remaining_in_regulation, home_goal_prob, simulations)
    home_goals = home_goals + initial_home_goals
    away_goals = np.random.binomial(seconds_remaining_in_regulation, away_goal_prob, simulations)
    away_goals = away_goals + initial_away_goals
    res = pd.DataFrame().assign(home_goals = home_goals, away_goals = away_goals)

    res = res.assign(outcome = np.where(res.home_goals > res.away_goals, 'Home Wins in Regulation', 'Away Wins in Regulation'))

    ot_outcomes = []

    for i in range(0, len(res[res.home_goals==away_goals])):
        
        ot_outcomes.append(simulate_regular_season_ot(home_goal_prob, away_goal_prob))

    ot = res[res.home_goals==away_goals].assign(outcome = ot_outcomes)

    ot = ot.assign(home_goals = np.where(ot.outcome=='Home Wins in OT', 1 + ot.home_goals, ot.home_goals),
             away_goals = np.where(ot.outcome=='Away Wins in OT', 1 + ot.away_goals, ot.away_goals))

    ot = ot.assign(home_goals = np.where(ot.outcome=='Home Wins in Shootout', 1 + ot.home_goals, ot.home_goals),
             away_goals = np.where(ot.outcome=='Away Wins in Shootout', 1 + ot.away_goals, ot.away_goals))
    
    final_results = pd.concat([
        res[res.home_goals!=away_goals],
        ot
    ]).reset_index(drop = True)
    
    final_results = final_results.assign(home_wins = np.where(
                                                            (final_results.home_goals>final_results.away_goals) | 
                                                            (final_results.outcome == 'Home Wins in Shootout')
                                                              , 1, 0))
    
    final_results = final_results.assign(away_wins = 1 - final_results.home_wins)

    return final_results

def live_games_route():
    try:
        start_time = time.time()
        
        def update_records(row):
            # Parse records
            home_w, home_l, home_otl = map(int, row['home_record'].split(' - '))
            away_w, away_l, away_otl = map(int, row['away_record'].split(' - '))
            
            home_wins = row['home_win_probability'] == 1
            
            if row['outcome'] == 'REG':
                # Regulation win
                if home_wins:
                    home_w += 1
                    away_l += 1
                else:
                    away_w += 1
                    home_l += 1
            elif row['outcome'] in ['OT', 'SO']:
                # OT/Shootout win
                if home_wins:
                    home_w += 1
                    away_otl += 1
                else:
                    away_w += 1
                    home_otl += 1
            
            # Update row
            row['home_record'] = f"{home_w} - {home_l} - {home_otl}"
            row['away_record'] = f"{away_w} - {away_l} - {away_otl}"
            
            return row

        def safe_json_get(json_data, key, default=None):
            """
            Safely extract a key from JSON data with multiple fallbacks
            """
            try:
                # Handle null/NaN values
                if pd.isna(json_data):
                    return default
                    
                # If it's already a dict, use it directly
                if isinstance(json_data, dict):
                    return json_data.get(key, default)
                    
                # If it's a string, try to parse as JSON
                if isinstance(json_data, str):
                    parsed = json.loads(json_data)
                    if isinstance(parsed, dict):
                        return parsed.get(key, default)
                        
                return default
                
            except (json.JSONDecodeError, TypeError, AttributeError):
                return default

        today_string = datetime.datetime.today().strftime('%Y-%m-%d')

        gp = pd.read_csv(f'{DATA_PATH}/Game_Projections_2025_2026.csv')

        today_gp = gp[(gp.date<=today_string) & (gp.state!='Final')]

        url = f"https://api-web.nhle.com/v1/schedule/{min(today_gp['date'])}"

        response = requests.get(url, timeout=500)
        data = response.json()

        finished_games = []

        for date_obj in data['gameWeek']:
            for game_obj in date_obj['games']:
                if game_obj['gameState'] in ['OFF', 'FINAL']:
                    finished_games.append(game_obj)

        finished_games_df = pd.DataFrame()

        for game in finished_games:
            game_id = game['id']
            print(game_id)
            this_game_df = today_gp[today_gp.ID==game_id]
            season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
            website_gid = str(game_id)[5:]
            away_score = game['awayTeam']['score']
            home_score = game['homeTeam']['score']
            period_type = game['periodDescriptor']['periodType']  

            if True:
                if home_score > away_score:
                    winner = this_game_df.home_team.iloc[0]
                    home_win_probability = 1
                    home_cup = this_game_df.home_cup_if_win.iloc[0]
                    home_playoffs = this_game_df.home_playoffs_if_win.iloc[0]
                    home_points = this_game_df.home_points_if_win.iloc[0]
                    if period_type == 'REG':
                        away_cup = this_game_df.away_cup_if_lose_regulation.iloc[0]
                        away_playoffs = this_game_df.away_playoffs_if_lose_regulation.iloc[0]
                        away_points = this_game_df.away_points_if_lose_regulation.iloc[0]
                    elif period_type in ['OT', 'SO']:
                        away_cup = this_game_df.away_cup_if_lose_overtime.iloc[0]
                        away_playoffs = this_game_df.away_playoffs_if_lose_overtime.iloc[0]
                        away_points = this_game_df.away_points_if_lose_overtime.iloc[0]
                    else:
                        print('ERROR: ISNT ACTUALLY OVER?')
                elif away_score > home_score:
                    winner = this_game_df.away_team.iloc[0]
                    home_win_probability = 0
                    away_cup = this_game_df.away_cup_if_win.iloc[0]
                    away_playoffs = this_game_df.away_playoffs_if_win.iloc[0]
                    away_points = this_game_df.away_points_if_win.iloc[0]
                    if period_type == 'REG':
                        home_cup = this_game_df.home_cup_if_lose_regulation.iloc[0]
                        home_playoffs = this_game_df.home_playoffs_if_lose_regulation.iloc[0]
                        home_points = this_game_df.home_points_if_lose_regulation.iloc[0]
                    elif period_type in ['OT', 'SO']:
                        home_cup = this_game_df.home_cup_if_lose_overtime.iloc[0]
                        home_playoffs = this_game_df.home_playoffs_if_lose_overtime.iloc[0]
                        home_points = this_game_df.home_points_if_lose_overtime.iloc[0]
                        
                else:
                    print('ERROR: DONT HAVE A WINNER')

                this_game_df = this_game_df.loc[:, 
                    ['ID', 'date', 'utc_start_time', 'home_team', 'away_team', 'home_record', 'away_record'
                    ]].assign(state = 'Final', 
                                outcome = period_type,
                                home_win_probability = home_win_probability,
                                projected_home_goals = home_score,
                                projected_away_goals = away_score,
                                home_cup = home_cup,
                                home_playoffs = home_playoffs,
                                home_points = home_points,
                                away_cup = away_cup,
                                away_playoffs = away_playoffs,
                                away_points = away_points,
                                home_goals = home_score,
                                away_goals = away_score)

                this_game_df = this_game_df.apply(update_records, axis = 1)

            finished_games_df = finished_games_df._append(this_game_df)

        live_games = []

        for date_obj in data['gameWeek']:
            for game_obj in date_obj['games']:
                if game_obj['gameState'] in ['LIVE', 'CRIT']:
                    live_games.append(game_obj)

        live_games_df = pd.DataFrame()

        for game in live_games:
            game_id = game['id']
            print(game_id)
            this_game_df = today_gp[today_gp.ID==game_id]
            season = str(game_id)[:4] + str(int(str(game_id)[:4]) + 1)
            website_gid = str(game_id)[5:]
            live_url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/landing'
            live_response = requests.get(live_url)
            live_data = live_response.json()
            initial_away_goals = game['awayTeam']['score']
            initial_home_goals = game['homeTeam']['score']
            period_type = game['periodDescriptor']['periodType'] 
            period = game['periodDescriptor']['number']
            time_remaining = live_data['clock']['timeRemaining']

            if period <= 3:
                if live_data['clock']['inIntermission'] == True:
                    remaining_seconds_in_regulation = 3600 - (1200 * period)
                if live_data['clock']['inIntermission'] == False:
                    remaining_seconds_in_regulation = 3600 - (1200 * period) + live_data['clock']['secondsRemaining']

                sim_df = simulate_live_game_in_regulation(game['homeTeam']['abbrev'], game['awayTeam']['abbrev'],
                                                    10000, initial_home_goals, initial_away_goals, 
                                                    remaining_seconds_in_regulation)

            elif period == 4:
                if live_data['clock']['inIntermission'] == True:
                    remaining_seconds_in_ot = 300
                if live_data['clock']['inIntermission'] == False:
                    remaining_seconds_in_ot = live_data['clock']['secondsRemaining']
                home_goal_prob, away_goal_prob = compute_home_goal_away_goal_probs(game['homeTeam']['abbrev'], game['awayTeam']['abbrev'])
                ot_outcomes = []

                for i in range(0, 10000):
                    
                    ot_outcomes.append(simulate_regular_season_ot(home_goal_prob, away_goal_prob, remaining_seconds_in_ot))

                ot = pd.DataFrame().assign(outcome = ot_outcomes)
                
                ot = ot.assign(home_goals = initial_home_goals, away_goals = initial_away_goals)

                ot = ot.assign(home_goals = np.where(ot.outcome=='Home Wins in OT', 1 + ot.home_goals, ot.home_goals),
                        away_goals = np.where(ot.outcome=='Away Wins in OT', 1 + ot.away_goals, ot.away_goals))

                ot = ot.assign(home_goals = np.where(ot.outcome=='Home Wins in Shootout', 1 + ot.home_goals, ot.home_goals),
                        away_goals = np.where(ot.outcome=='Away Wins in Shootout', 1 + ot.away_goals, ot.away_goals))
                
                sim_df = ot.assign(home_wins = np.where((ot.home_goals>ot.away_goals) | 
                                                            (ot.outcome == 'Home Wins in Shootout')
                                                            , 1, 0))
                
            elif period == 5:
                home_shootout_wins = pd.DataFrame({
                    'outcome': ['Home Wins in Shootout'] * 5000,
                    'home_goals': [initial_home_goals + 1] * 5000,
                    'away_goals': [initial_away_goals] * 5000,
                    'home_wins': [1] * 5000
                })

                # Create 5K records for Away Wins in Shootout
                away_shootout_wins = pd.DataFrame({
                    'outcome': ['Away Wins in Shootout'] * 5000,
                    'home_goals': [initial_home_goals] * 5000,
                    'away_goals': [initial_away_goals + 1] * 5000,
                    'home_wins': [0] * 5000
                })

                # Combine into single dataframe
                sim_df = pd.concat([home_shootout_wins, away_shootout_wins], ignore_index=True)

            this_game_df = this_game_df.drop(columns = {'home_win_probability', 'projected_home_goals', 'projected_away_goals', 'state'}).assign(
                home_goals = initial_home_goals,
                away_goals = initial_away_goals,
                home_win_probability = sim_df.home_wins.mean().item(),
                projected_home_goals = sim_df.home_goals.mean().item(),
                projected_away_goals = sim_df.away_goals.mean().item(),
                state = 'In Progress',
                period = period,
                in_intermission = live_data['clock']['inIntermission'],
                time_remaining = time_remaining
            ).loc[:, ['ID', 'date', 'utc_start_time', 'home_team', 'away_team', 'home_record', 'away_record', 'state', 
                        'home_win_probability', 'projected_home_goals', 'projected_away_goals', 'period', 'time_remaining',
                        'in_intermission', 'home_goals', 'away_goals', 'home_playoffs_if_win', 'home_cup_if_win', 'home_points_if_win', 'home_playoffs_if_lose_overtime', 'home_cup_if_lose_overtime', 'home_points_if_lose_overtime',
                        'home_playoffs_if_lose_regulation', 'home_cup_if_lose_regulation', 'home_points_if_lose_regulation', 'away_playoffs_if_win', 'away_cup_if_win', 'away_points_if_win',
                        'away_playoffs_if_lose_overtime', 'away_cup_if_lose_overtime', 'away_points_if_lose_overtime', 'away_playoffs_if_lose_regulation', 'away_cup_if_lose_regulation',
                        'away_points_if_lose_regulation']]

            live_games_df = live_games_df._append(this_game_df)

        final_via_csv = gp[(~gp.ID.isin([game['id'] for game in finished_games])) & (~gp.ID.isin([game['id'] for game in live_games])) & (gp.state=='Final')]
        scheduled_via_csv = gp[(~gp.ID.isin([game['id'] for game in finished_games])) & (~gp.ID.isin([game['id'] for game in live_games])) & (gp.state.isin(['Scheduled', 'Pre-Game', 'In Progress']))]

        final_via_csv = final_via_csv.loc[:, ['ID', 'date', 'utc_start_time', 'state', 'home_team', 'away_team', 'home_win_probability', 'projected_home_goals', 'projected_away_goals',
                            'home_playoffs_if_win', 'home_cup_if_win', 'home_points_if_win', 'home_playoffs_if_lose_overtime', 'home_cup_if_lose_overtime', 'home_points_if_lose_overtime',
                            'home_playoffs_if_lose_regulation', 'home_cup_if_lose_regulation', 'home_points_if_lose_regulation', 'away_playoffs_if_win', 'away_cup_if_win', 'away_points_if_win',
                            'away_playoffs_if_lose_overtime', 'away_cup_if_lose_overtime', 'away_points_if_lose_overtime', 'away_playoffs_if_lose_regulation', 'away_cup_if_lose_regulation',
                            'away_points_if_lose_regulation', 'home_record', 'away_record']]

        scheduled_via_csv = scheduled_via_csv.loc[:, ['ID', 'date', 'utc_start_time', 'state', 'home_team', 'away_team', 'home_win_probability', 'projected_home_goals', 'projected_away_goals',
                            'home_playoffs_if_win', 'home_cup_if_win', 'home_points_if_win', 'home_playoffs_if_lose_overtime', 'home_cup_if_lose_overtime', 'home_points_if_lose_overtime',
                            'home_playoffs_if_lose_regulation', 'home_cup_if_lose_regulation', 'home_points_if_lose_regulation', 'away_playoffs_if_win', 'away_cup_if_win', 'away_points_if_win',
                            'away_playoffs_if_lose_overtime', 'away_cup_if_lose_overtime', 'away_points_if_lose_overtime', 'away_playoffs_if_lose_regulation', 'away_cup_if_lose_regulation',
                            'away_points_if_lose_regulation', 'home_record', 'away_record']]

        all_gp_with_updates = pd.concat([
                                    final_via_csv,
                                    scheduled_via_csv,
                                    finished_games_df,
                                    live_games_df
                                ]).sort_values(by = 'ID', ascending = True)

        live_games_api_data = json.loads(all_gp_with_updates.to_json(orient = 'records'))

        response_data = {'success': True, 'games': live_games_api_data}
        response = make_response(response_data)
        response.headers['Cache-Control'] = 'public, s-maxage=10, stale-while-revalidate=60'
        return response

    except Exception as e:
        print(f"Error in live-games-python: {e}")
        error_msg = f"ERROR in live-games-python:\n"
        error_msg = f"Exception: {type(e).__name__}: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        response_data = {'success': False, 'games': []}
        response = make_response(response_data)
        return response

