from flask import make_response, request
import pandas as pd
import requests
import datetime
import os
import numpy as np
import json
import onnxruntime as rt
from TopDownHockey_Scraper.TopDownHockey_NHL_Scraper import full_scrape
import traceback
import sys

all_strength_states = ['3v3', '4v4', '5v5', '5v4', '4v5', '5v3', '3v5', '4v3', '3v4', '5vE', 'Ev5', '4vE', 'Ev4', '3vE', 'Ev3']

def live_games_pbp_route():

    try:

        game_id = request.args.get('game_id')
        home_team = request.args.get('home_team')
        away_team = request.args.get('away_team')
        
        print(f"Query parameters received - game_id: {game_id}, home_team: {home_team}, away_team: {away_team}")
        
        fenwick = ('SHOT', 'MISS', 'GOAL')
        corsi = ('SHOT', 'MISS', 'BLOCK', 'GOAL')
        even_strength = ('3v3', '4v4', '5v5')
        power_play = ('5v4', '4v5', '5v3', '3v5', '4v3', '3v4')
        home_power_play = ('5v4', '5v3', '4v3')
        away_power_play = ('4v5', '3v5', '3v4')
        empty_net = ('5vE', 'Ev5', '4vE', 'Ev4', '3vE', 'Ev3')
        not_distance = ('Tip-In', 'Wrap-around', 'Deflected')
        meaningful = ('FAC', 'GOAL', 'BLOCK', 'SHOT', 'MISS', 'HIT', 'TAKE', 'GIVE')
        stoppages = ('PGSTR', 'PGEND', 'ANTHEM', 'PSTR', 'FAC', 'GOAL', 'STOP', 'PENL', 'PEND', 'CHL', 'GEND', 'GOFF', 'EISTR', 'EIEND', 'EGT', 'EGPID')

        # Determine model path based on environment
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(script_dir, 'models')

        # Load ONNX models
        EV_model = rt.InferenceSession(os.path.join(MODEL_PATH, "EV_2022_2023_xG.onnx"))
        PP_model = rt.InferenceSession(os.path.join(MODEL_PATH, "PP_2022_2023_xG.onnx"))
        SH_model = rt.InferenceSession(os.path.join(MODEL_PATH, "SH_2022_2023_xG.onnx"))
        EN_model = rt.InferenceSession(os.path.join(MODEL_PATH, "EN_2022_2023_xG.onnx"))

        def prepare_fenwicks(df, choice, playoffs = False):
            # CLEAN DATA #

            ## FIX EVENT ZONES ##

            df['event_zone'] = np.where((df['event_zone'] == 'Def') & (df['event_type'] == 'BLOCK'), 'Off', df['event_zone'])

            ### FILTER OUT DATAFRAME WITH MEANINGFUL EVENTS ###

            if playoffs == False:

                filtered_data = df.loc[((df.event_type.isin(meaningful)) & (df.game_period<5) & 
                                    (~pd.isna(df.coords_x)) & (~pd.isna(df.coords_y))) #& ([bool(re.search("Penalty Shot", i)) for i in df.event_description]==False)
                                        ]

            else:

                filtered_data = df.loc[((df.event_type.isin(meaningful)) & 
                                    (~pd.isna(df.coords_x)) & (~pd.isna(df.coords_y))) #& ([bool(re.search("Penalty Shot", i)) for i in df.event_description]==False)
                                        ]
            
            filtered_data = filtered_data.loc[~(filtered_data.event_type.isin(corsi) & ((filtered_data.coords_x)==0) & ((filtered_data.coords_y)==0))]

            ### ADD OUR FIRST SET OF CONTEXTUAL VARIABLES ###

            filtered_data = filtered_data.assign(
            distance = np.sqrt((89 - abs(filtered_data.coords_x))**2 + filtered_data.coords_y**2)
            ).assign(
            angle = abs(np.arctan(filtered_data.coords_y / (89 - abs(filtered_data.coords_x))) * (180 / np.pi))
            ).assign(
            last_team = np.where(filtered_data.event_team.shift()==filtered_data.event_team, "same", "diff")
            ).assign(
            is_home = np.where(filtered_data.event_team==filtered_data.home_team, 1, 0)
            ).assign(
            last_event = filtered_data.event_type.shift()
            ).assign(
            time_diff = filtered_data.game_seconds - filtered_data.game_seconds.shift()
            ).assign(
            last_coords_x = filtered_data.coords_x.shift()
            ).assign(
            last_coords_y = filtered_data.coords_y.shift()
            ).assign(
            distance_change = np.sqrt((filtered_data.coords_x - filtered_data.coords_x.shift())**2 + (filtered_data.coords_y - filtered_data.coords_y.shift())**2)
            ).assign(
            speed_distance_change = np.where(filtered_data.game_seconds - filtered_data.game_seconds.shift()==0,
            np.sqrt((filtered_data.coords_x - filtered_data.coords_x.shift())**2 + (filtered_data.coords_y - filtered_data.coords_y.shift())**2)/2,
            np.sqrt((filtered_data.coords_x - filtered_data.coords_x.shift())**2 + (filtered_data.coords_y - filtered_data.coords_y.shift())**2)/
                (filtered_data.game_seconds - filtered_data.game_seconds.shift()))
            ).assign(
            reported_distance = np.where(filtered_data.event_type.isin(fenwick),
                                        (filtered_data.event_description.replace('.*Zone, *(.*?) * ft.*','\\1',regex=True, inplace = False)),
                                        np.nan)
            )

            filtered_data.reported_distance = (
                pd.to_numeric(filtered_data.reported_distance.astype(str).str.replace(',',''), errors='coerce')
                    .fillna(0)
                    .astype(int)
            )

            filtered_data.distance = np.where((filtered_data.event_type.isin(fenwick)) & (filtered_data.reported_distance>89) &
                                            (filtered_data.coords_x<0) & ~(filtered_data.event_detail.isin(not_distance)) & (filtered_data.event_zone!="Off"), 
                                            np.sqrt((abs(filtered_data.coords_x) + 89)**2 + filtered_data.coords_y**2), 
                                            np.where(filtered_data.event_type.isin(fenwick) & (filtered_data.reported_distance>89) &
                                                    (filtered_data.coords_x>0) & ~(filtered_data.event_detail.isin(not_distance)) & (filtered_data.event_zone!="Off"), 
                                                    np.sqrt((filtered_data.coords_x + 89)**2 + filtered_data.coords_y**2),
                                                    filtered_data.distance))

            filtered_data.angle = np.where((filtered_data.event_type.isin(fenwick)) & (filtered_data.reported_distance>89) &
                                            (filtered_data.coords_x<0) & ~(filtered_data.event_detail.isin(not_distance)) & (filtered_data.event_zone!="Off"), 
                                            abs(np.arctan(filtered_data.coords_y/(abs(filtered_data.coords_x) + 89)) * (180 / np.pi)), 
                                            np.where(filtered_data.event_type.isin(fenwick) & (filtered_data.reported_distance>89) &
                                                    (filtered_data.coords_x>0) & ~(filtered_data.event_detail.isin(not_distance)) & (filtered_data.event_zone!="Off"), 
                                                    abs(np.arctan(filtered_data.coords_y / (filtered_data.coords_x + 89)) * (180 / np.pi)),
                                                    filtered_data.angle))
            
            shots = filtered_data.loc[(filtered_data.event_type.isin(fenwick))]
            
            shots = shots.assign(
            goal = np.where(shots.event_type=="GOAL", 1, 0)
            ).assign(
            prior_FAC = np.where(shots.last_event=="FAC", 1, 0)
            ).assign(
            prior_same_HIT = np.where((shots.last_team=="same") & (shots.last_event=="HIT"), 1, 0)
            ).assign(
            prior_same_SHOT = np.where((shots.last_team=="same") & (shots.last_event=="SHOT"), 1, 0)
            ).assign(
            prior_same_MISS = np.where((shots.last_team=="same") & (shots.last_event=="MISS"), 1, 0)
            ).assign(
            prior_same_TAKE = np.where((shots.last_team=="same") & (shots.last_event=="TAKE"), 1, 0)
            ).assign(
            prior_same_GIVE = np.where((shots.last_team=="same") & (shots.last_event=="GIVE"), 1, 0)
            ).assign(
            prior_same_BLOCK = np.where((shots.last_team=="same") & (shots.last_event=="BLOCK"), 1, 0)
            ).assign(
            prior_diff_HIT = np.where((shots.last_team=="diff") & (shots.last_event=="HIT"), 1, 0)
            ).assign(
            prior_diff_SHOT = np.where((shots.last_team=="diff") & (shots.last_event=="SHOT"), 1, 0)
            ).assign(
            prior_diff_MISS = np.where((shots.last_team=="diff") & (shots.last_event=="MISS"), 1, 0)
            ).assign(
            prior_diff_TAKE = np.where((shots.last_team=="diff") & (shots.last_event=="TAKE"), 1, 0)
            ).assign(
            prior_diff_GIVE = np.where((shots.last_team=="diff") & (shots.last_event=="GIVE"), 1, 0)
            ).assign(
            prior_diff_BLOCK = np.where((shots.last_team=="diff") & (shots.last_event=="BLOCK"), 1, 0)
            ).assign(
            tied = np.where(shots.home_score==shots.away_score, 1, 0)
            ).assign(
            up_1 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)==1)) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)==1)), 1, 0)
            ).assign(
            up_2 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)==2)) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)==2)), 1, 0)
            ).assign(
            up_3 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)>2)) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)>2)), 1, 0)
            ).assign(
            down_1 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)==(-1))) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)==(-1))), 1, 0)
            ).assign(
            down_2 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)==(-2))) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)==(-2))), 1, 0)
            ).assign(
            down_3 = np.where(((shots.event_team==shots.home_team) & ((shots.home_score-shots.away_score)<(-2))) |
                        ((shots.event_team==shots.away_team) & ((shots.away_score-shots.home_score)<(-2))), 1, 0)
            ).assign(
            wrist = np.where((shots.event_detail=="Wrist") | (pd.isna(shots.event_detail)), 1, 0)
            ).assign(
            backhand = np.where(shots.event_detail=="Backhand", 1, 0)
            ).assign(
            snap = np.where(shots.event_detail=="Snap", 1, 0)
            ).assign(
            slap = np.where(shots.event_detail=="Slap", 1, 0)
            ).assign(
            tip = np.where(shots.event_detail=="Tip", 1, 0)
            ).assign(
            wrap = np.where(shots.event_detail=="Wrap-around", 1, 0)
            ).assign(
            deflected = np.where(shots.event_detail=="Deflected", 1, 0)
            ).assign(
            PS = np.where(shots.event_detail=="Penalty Shot", 1, 0)).assign(
            five = np.where((shots.game_strength_state=="5v5") | 
                ((shots.home_skaters==5) & (shots.away_skaters==5) & (shots.game_strength_state=="Ev5") & (shots.event_team==shots.home_team)) |
                ((shots.home_skaters==5) & (shots.away_skaters==5) & (shots.game_strength_state=="5vE") & (shots.event_team==shots.away_team)),
                            1, 0)
            ).assign(
            four = np.where((shots.game_strength_state=="4v4") | 
                ((shots.home_skaters==4) & (shots.away_skaters==4) & (shots.game_strength_state=="Ev4") & (shots.event_team==shots.home_team)) |
                ((shots.home_skaters==4) & (shots.away_skaters==4) & (shots.game_strength_state=="4vE") & (shots.event_team==shots.away_team)),
                            1, 0)
            ).assign(
            three = np.where((shots.game_strength_state=="3v3") | 
                ((shots.home_skaters==3) & (shots.away_skaters==3) & (shots.game_strength_state=="Ev3") & (shots.event_team==shots.home_team)) |
                ((shots.home_skaters==3) & (shots.away_skaters==3) & (shots.game_strength_state=="3vE") & (shots.event_team==shots.away_team)),
                            1, 0)
            ).assign(
            off_EN = np.where(((shots.event_team==shots.home_team) & (shots.away_goalie=='\xa0')) |
                            ((shots.event_team==shots.away_team) & (shots.home_goalie=='\xa0')),
                            1, 0)
            )
            
            shots = shots.assign(
            off_5v4 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="5v4")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="4v5")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==5) & (shots.away_skaters==4) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==4) & (shots.away_skaters==5) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_5v3 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="5v3")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="3v5")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==5) & (shots.away_skaters==3) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==3) & (shots.away_skaters==5) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_4v3 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="4v3")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="3v4")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==4) & (shots.away_skaters==3) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==3) & (shots.away_skaters==4) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_4v5 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="4v5")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="5v4")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==4) & (shots.away_skaters==5) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==5) & (shots.away_skaters==4) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_3v5 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="3v5")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="5v3")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==3) & (shots.away_skaters==5) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==5) & (shots.away_skaters==3) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_3v4 = np.where(((shots.event_team==shots.home_team) & (shots.game_strength_state=="3v4")) |
                            ((shots.event_team==shots.away_team) & (shots.game_strength_state=="4v3")) |
                            ((shots.event_team==shots.home_team) & (shots.home_skaters==3) & (shots.away_skaters==4) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==4) & (shots.away_skaters==3) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_6v5 = np.where(((shots.event_team==shots.home_team) & (shots.home_skaters==6) & (shots.away_skaters==5) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==5) & (shots.away_skaters==6) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_6v4 = np.where(((shots.event_team==shots.home_team) & (shots.home_skaters==6) & (shots.away_skaters==4) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==4) & (shots.away_skaters==6) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            off_6v3 = np.where(((shots.event_team==shots.home_team) & (shots.home_skaters==6) & (shots.away_skaters==3) & (shots.off_EN==0)) |
                            ((shots.event_team==shots.away_team) & (shots.home_skaters==3) & (shots.away_skaters==6) & (shots.off_EN==0)),
                            1, 0)
            ).assign(
            no_prior = np.where(((shots.prior_FAC==0) & 
                                (shots.prior_same_HIT==0) & (shots.prior_same_SHOT==0) & (shots.prior_same_MISS==0) & (shots.prior_same_TAKE==0) & (shots.prior_same_GIVE==0) & (shots.prior_same_BLOCK==0) &
                                (shots.prior_diff_HIT==0) & (shots.prior_diff_SHOT==0) & (shots.prior_diff_MISS==0) & (shots.prior_diff_TAKE==0) & (shots.prior_diff_GIVE==0) & (shots.prior_diff_BLOCK==0)
                                ), 1, 0)
            ).assign(
            btn = np.where((abs(shots.coords_x)>89) & (shots.reported_distance<89), 1, 0)
            )
            
            shots = shots[~shots.event_description.str.contains("Penalty Shot")]
            global backup_shots 
            backup_shots = shots
            
            shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected', 'five', 'four', 'three', 
                                'off_EN', 'off_5v4', 'off_5v3', 'off_4v3', 'off_4v5','off_3v5', 'off_3v4', 'off_6v5', 'off_6v4', 'off_6v3', 'no_prior', 'home_team', 'away_team', 'coordinate_source', 'btn']]
            
            EV_shots = shots.loc[(shots.five==1) | (shots.four==1) | (shots.three==1)]
            EV_shots = EV_shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected', 'four', 'three', 'no_prior', 'btn']]
            PP_shots = shots.loc[(shots.off_5v4==1) | (shots.off_5v3==1) | (shots.off_4v3==1) | (shots.off_6v5==1) | (shots.off_6v4==1) | (shots.off_6v3==1)]
            PP_shots = PP_shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                'off_5v4', 'off_5v3', 'off_4v3','off_6v5', 'off_6v4', 'off_6v3', 'no_prior', 'btn']]
            SH_shots = shots.loc[(shots.off_4v5==1) | (shots.off_3v5==1) | (shots.off_3v4==1)]
            SH_shots = SH_shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                'off_4v5', 'off_3v5', 'off_3v4', 'no_prior', 'btn']]
            EN_shots = shots.loc[(shots.off_EN==1)]
            EN_shots = EN_shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                'off_EN', 'no_prior', 'btn']]
            
            if choice == "EV":
                return(EV_shots)
            elif choice == "PP":
                return(PP_shots)
            elif choice == "SH":
                return(SH_shots)
            elif choice == "EN":
                return(EN_shots)
            elif choice =="AS":
                return(shots)
            else:
                return("Whoops! Looks like you didn't enter a proper shot type. You have four choices: EV, PP, SH, and PP. Please enter one of these, with quotation marks surrounding it.")

        def get_target_fenwicks(choice, target_season_list, shots):
            
            shots = shots[shots.season.isin(target_season_list)]
            
            shots = shots[~((shots.five==0) & (shots.four==0) & (shots.three==0) & (shots.off_EN==0) & (shots.off_5v4==0) & (shots.off_5v3==0) & (shots.off_4v3==0) & (shots.off_4v5==0) & (shots.off_3v5==0) & 
            (shots.off_3v4==0) & (shots.off_6v5==0) & (shots.off_6v4==0) & (shots.off_6v3==0))]

            shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected', 'five', 'four', 'three', 
                                'off_EN', 'off_5v4', 'off_5v3', 'off_4v3', 'off_4v5','off_3v5', 'off_3v4', 'off_6v5', 'off_6v4', 'off_6v3', 'no_prior', 'home_team', 'away_team', 'btn']]

            if choice == 'EV':
            
                shots = shots.loc[(shots.five==1) | (shots.four==1) | (shots.three==1)]
                shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                    'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                    'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                    'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                    'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected', 'four', 'three', 'no_prior', 'btn']]
                
            if choice == 'PP': 
            
                shots = shots.loc[(shots.off_5v4==1) | (shots.off_5v3==1) | (shots.off_4v3==1) | (shots.off_6v5==1) | (shots.off_6v4==1) | (shots.off_6v3==1)]
                shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                    'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                    'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                    'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                    'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                    'off_5v3', 'off_4v3','off_6v5', 'off_6v4', 'off_6v3', 'no_prior', 'btn']]
                
            if choice == 'SH':
            
                shots = shots.loc[(shots.off_4v5==1) | (shots.off_3v5==1) | (shots.off_3v4==1)]
                shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                    'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                    'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                    'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                    'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                    'off_3v5', 'off_3v4', 'no_prior', 'btn']]
            
            if choice == 'EN':
            
                shots = shots.loc[(shots.off_EN==1)]
                shots = shots.loc[:, ['goal', 'game_id', 'event_index', 'event_player_1', 'is_home', 'distance', 'angle', 
                                    'coords_x', 'coords_y', 'last_coords_x', 'last_coords_y', 'distance_change', 'speed_distance_change', 
                                    'prior_FAC', 'prior_same_HIT', 'prior_same_SHOT', 'prior_same_MISS', 'prior_same_TAKE', 'prior_same_GIVE', 'prior_same_BLOCK',
                                    'prior_diff_HIT', 'prior_diff_SHOT', 'prior_diff_MISS', 'prior_diff_TAKE', 'prior_diff_GIVE', 'prior_diff_BLOCK',
                                    'up_1', 'up_2', 'up_3', 'down_1', 'down_2', 'down_3', 'wrist', 'backhand', 'snap', 'slap', 'tip', 'wrap', 'deflected',
                                    'no_prior', 'btn']]
                
            if choice == 'AS':
                
                shots = shots#.drop(columns = ['home_team', 'away_team', 'coordinate_source', 'season'])

            return shots

        def calculate_xg(game_strength, pbp_df, df):
            if game_strength=='SH':
                model = SH_model
            elif game_strength == 'PP':
                model = PP_model
            elif game_strength=='EV':
                model = EV_model
            elif game_strength=='EN':
                model = EN_model
            fenwicks = get_target_fenwicks(game_strength, [pbp_df.iloc[0].season], df.assign(season = pbp_df.iloc[0].season))
            if 'xG' in fenwicks.columns:
                fenwicks = fenwicks.drop(columns = ['xG'])
            if 'xG' in pbp_df.columns:
                pbp_df = pbp_df.drop(columns = ['xG'])
            if game_strength == 'EV':
                # TODO: Check if you put five and tied in the right order.
                fenwicks = fenwicks.assign(five = 1 - fenwicks.four - fenwicks.three)
                fenwicks = fenwicks.assign(tied = 1 - fenwicks.up_1 - fenwicks.up_2 - fenwicks.up_3 - fenwicks.down_1 - fenwicks.down_2 - fenwicks.down_3)
            shots = fenwicks.drop(columns = ['game_id', 'event_index', 'event_player_1'])
            outcomes = shots[shots.columns[0]]
            predictors = shots[shots.columns[1:(len(shots.columns))]]
            print('Predictors has this shape for game_strength: ' + game_strength)
            print(predictors.shape)
            
            # Convert to numpy array with float32 for ONNX
            input_data = predictors.values.astype(np.float32)
            input_name = model.get_inputs()[0].name
            
            # Run ONNX inference
            # Get all output names to request specific outputs
            output_names = [output.name for output in model.get_outputs()]
            raw_output = model.run(output_names, {input_name: input_data})
            
            # Debug output structure (only for EV to avoid spam)
            if game_strength == 'EV':
                print(f"\nDEBUG - Output structure:")
                print(f"  Number of outputs: {len(raw_output)}")
                print(f"  Output names: {output_names}")
                for i, (name, out) in enumerate(zip(output_names, raw_output)):
                    print(f"  Output {i} ({name}): shape={out.shape}, dtype={out.dtype}")
                    if out.size > 0:
                        print(f"    First 3 values: {out.flat[:3]}")
                        print(f"    Min/Max: {out.min()}/{out.max()}")
            
            # Extract predictions based on output structure
            # For XGBoost regressors converted to ONNX, typically:
            # - Single output: predictions as (n_samples, 1) or (n_samples,)
            # - Multiple outputs: (labels, scores) where scores are the predictions
            if len(raw_output) == 1:
                predictions = raw_output[0]
                # Flatten if 2D with single column
                if len(predictions.shape) == 2 and predictions.shape[1] == 1:
                    predictions = predictions[:, 0]
            else:
                # Multiple outputs - usually the last one has the scores
                predictions = raw_output[-1]
                # For dict-like output, extract the score column
                if len(predictions.shape) == 2:
                    # If it's a 2D array, take the appropriate column
                    if predictions.shape[1] == 1:
                        predictions = predictions[:, 0]
                    else:
                        # For binary classification converted from regression, take second column
                        predictions = predictions[:, -1]
            
            fenwicks = fenwicks.assign(xG = predictions)
            #print("This Year's AUC: " + str(round(sklearn.metrics.roc_auc_score(fenwicks.goal, fenwicks.xG), 4)))
            #print('Expected Goals per Goal: ' + str(round(((sum(fenwicks.xG)/sum(fenwicks.goal))), 4)))
            return fenwicks

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
            # Running locally - use data directory in project root
            DATA_PATH = os.path.join(script_dir, 'data')
        
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

        event_mapping = {
            'period-start': 'PSTR',
            'faceoff': 'FAC', 
            'penalty': 'PENL',
            'shot-on-goal': 'SHOT',
            'stoppage': 'STOP',
            'blocked-shot': 'BLOCK',
            'missed-shot': 'MISS',
            'hit': 'HIT',
            'delayed-penalty': 'DELPEN',
            'goal': 'GOAL',
            'takeaway': 'TAKE',
            'giveaway': 'GIVE',
            'period-end': 'PEND'
        }

        shot_type_mapping = {'tip-in': 'Tip-In',
            'wrist': 'Wrist',
            'snap': 'Snap',
            'backhand': 'Backhand',
            'slap': 'Slap',
            'deflected': 'Deflected',
            'poke': 'Poke',
            'wrap-around': 'Wrap-around',
            'between-legs': 'Between Legs',
            'wrap around': 'Wrap-around',
            'between legs': 'Between Legs',
            'wraparound': 'Wrap-around',
            'betweenlegs': 'Between Legs',
            'bat': 'Bat'
        }

        teams = pd.read_csv(f'{DATA_PATH}/Team_Projected_Data.csv')
        intercepts = pd.read_csv(f'{DATA_PATH}/Projected_Intercepts.csv')

        player_id_mapping = pd.read_csv(f'{DATA_PATH}/portrait_links.csv').set_index('id')['player'].to_dict()

        gp = pd.read_csv(f'{DATA_PATH}/Game_Projections_2025_2026.csv')

        home_ff_int = intercepts['home_ff_int'].iloc[0]
        home_gf_int = intercepts['home_gf_int'].iloc[0]
        away_ff_int = intercepts['away_ff_int'].iloc[0]
        away_gf_int = intercepts['away_gf_int'].iloc[0]

        today_string = datetime.datetime.today().strftime('%Y-%m-%d')

        today_gp = gp[(gp.date<=today_string) & (gp.state!='Final')]

        today_gp = today_gp[today_gp.ID.astype(int)==int(game_id)]

        url = f"https://api-web.nhle.com/v1/schedule/{min(today_gp['date'])}"

        response = requests.get(url, timeout=500)
        data = response.json()

        finished_games = []
        game_is_finished = False  # Track if requested game is finished

        for date_obj in data['gameWeek']:
            for game_obj in date_obj['games']:
                if game_obj['gameState'] in ['OFF', 'FINAL']:
                    finished_games.append(game_obj)
                    if str(game_obj['id']) == str(game_id):
                        game_is_finished = True

        finished_game_pbps = pd.DataFrame()

        for game in finished_games:
            if str(game['id']) == str(game_id):
                print(game_id)
                try:
                    finished_game_pbp = full_scrape([int(game_id)])
                except Exception as e:
                    print(f"Error scraping finished game pbp for game {game_id}: {e}")
                    error_msg = f"ERROR in live-games-pbp-python:\n"
                    error_msg += f"Exception: {type(e).__name__}: {str(e)}\n"
                    error_msg += f"Traceback:\n{traceback.format_exc()}"
                    print(error_msg, file=sys.stderr)
                    continue
                print('Successfully scraped finished game pbp for game:', game_id)
                finished_game_pbps = finished_game_pbps._append(finished_game_pbp)

        live_games = []

        for date_obj in data['gameWeek']:
            for game_obj in date_obj['games']:
                if game_obj['gameState'] in ['LIVE', 'CRIT']:
                    live_games.append(game_obj)

        live_games_df = pd.DataFrame()
        live_games_fenwicks_df = pd.DataFrame()

        live_game_pbps = pd.DataFrame()

        for game in live_games:
            if str(game['id']) == str(game_id):
                print(game_id)
                try:
                    live_game_pbp = full_scrape([int(game_id)])
                except Exception as e:
                    print(f"Error scraping live game pbp for game {game_id}: {e}")
                    error_msg = f"ERROR in live-games-pbp-python:\n"
                    error_msg += f"Exception: {type(e).__name__}: {str(e)}\n"
                    error_msg += f"Traceback:\n{traceback.format_exc()}"
                    print(error_msg, file=sys.stderr)
                    continue
                print('Successfully scraped live game pbp for game:', game_id)
                live_game_pbps = live_game_pbps._append(live_game_pbp)

        print(finished_game_pbps)
        print(live_game_pbps)

        pbps = pd.concat([finished_game_pbps, live_game_pbps])

        if len(pbps) > 0:
            # Add season column (inferred from game_date)
            pbps['season'] = pbps['game_date'].astype(str).str[:4].astype(int)

            pbps = pbps.assign(home_team = np.where(pbps.home_team=='LAK', 'L.A', pbps.home_team),
                    away_team = np.where(pbps.away_team=='LAK', 'L.A', pbps.away_team),
                    event_team = np.where(pbps.event_team=='LAK', 'L.A', pbps.event_team))

            pbps = pbps.assign(home_team = np.where(pbps.home_team=='NJD', 'N.J', pbps.home_team),
                        away_team = np.where(pbps.away_team=='NJD', 'N.J', pbps.away_team),
                        event_team = np.where(pbps.event_team=='NJD', 'N.J', pbps.event_team))

            pbps = pbps.assign(home_team = np.where(pbps.home_team=='TBL', 'T.B', pbps.home_team),
                        away_team = np.where(pbps.away_team=='TBL', 'T.B', pbps.away_team),
                        event_team = np.where(pbps.event_team=='TBL', 'T.B', pbps.event_team))

            pbps = pbps.assign(home_team = np.where(pbps.home_team=='SJS', 'S.J', pbps.home_team),
                        away_team = np.where(pbps.away_team=='SJS', 'S.J', pbps.away_team),
                        event_team = np.where(pbps.event_team=='SJS', 'S.J', pbps.event_team))

            sb2023 = pd.read_csv(f'{DATA_PATH}/Scorekeeper_Bias_2024.csv')

            df = prepare_fenwicks(pbps, "AS").merge(sb2023).drop(columns = ['average_home_distance', 'average_away_distance'])

            df = df.assign(distance = df['distance'] + df['scorekeeper_bias_adjustment']).drop(columns = 'scorekeeper_bias_adjustment')
            df['distance'] = np.where(df['distance']>188, 188, df['distance'])
            df['distance'] = np.where(df['distance']<1, 1, df['distance'])

            sh_fenwicks = calculate_xg('SH', pbps, df)
            ev_fenwicks = calculate_xg('EV', pbps, df)
            pp_fenwicks = calculate_xg('PP', pbps, df)
            en_fenwicks = calculate_xg('EN', pbps, df)

            pbp_with_xg = pbps.merge(pd.concat([
                sh_fenwicks.loc[:, ['game_id', 'event_index', 'xG']],
                ev_fenwicks.loc[:, ['game_id', 'event_index', 'xG']],
                pp_fenwicks.loc[:, ['game_id', 'event_index', 'xG']],
                en_fenwicks.loc[:, ['game_id', 'event_index', 'xG']]
            ]), how = 'left')

            pbp_with_xg = pbp_with_xg[pbp_with_xg.event_type.isin(['SHOT', 'BLOCK', 'GOAL', 'MISS'])]

            pbp_with_xg = pbp_with_xg.assign(goalie_against = np.where(pbp_with_xg.event_team == pbp_with_xg.home_team, pbp_with_xg.away_goalie, pbp_with_xg.home_goalie))

            pbp_with_xg = pbp_with_xg.rename(columns = {'coords_x': 'x', 'coords_y': 'y', 'event_player_1':'shooter', 'event_team':'team'})

            pbp_with_xg = pbp_with_xg.assign(originalX = np.where(pbp_with_xg.game_period.isin([2, 4]), 1 * pbp_with_xg.x, pbp_with_xg.x),
                                    originalY = np.where(pbp_with_xg.game_period.isin([2, 4]), 1 * pbp_with_xg.y, pbp_with_xg.y))

            pbp_with_xg = pbp_with_xg.assign(x = np.where(pbp_with_xg.game_period.isin([2, 4]), -1 * pbp_with_xg.x, pbp_with_xg.x),
                                    y = np.where(pbp_with_xg.game_period.isin([2, 4]), -1 * pbp_with_xg.y, pbp_with_xg.y))

            if len(pbp_with_xg) > 10 and pbp_with_xg[(pbp_with_xg.game_period==1) & (pbp_with_xg.team==pbp_with_xg.home_team)].x.mean().item() > 0:
                print('Flipping x and y for game id: ' + str(game_id))
                pbp_with_xg.x = -1 * pbp_with_xg.x
                pbp_with_xg.y = -1 * pbp_with_xg.y

            pbp_with_xg.x = -1 * pbp_with_xg.x
            pbp_with_xg.y = -1 * pbp_with_xg.y

            pbp_with_xg = pbp_with_xg[~((pbp_with_xg.x == 0) & (pbp_with_xg.y == 0))]

            pbp_with_xg = pbp_with_xg[~pd.isna(pbp_with_xg.x)]
            pbp_with_xg = pbp_with_xg[~pd.isna(pbp_with_xg.y)]

            pbp_with_xg.game_date = pbp_with_xg.game_date.astype(str)

            pbp_with_xg = pbp_with_xg.loc[:, ['x', 'y', 'team', 'game_id', 'game_date', 'game_period', 'game_seconds', 'shooter', 'goalie_against', 'event_type', 'event_detail', 'originalX', 'originalY', 'xG', 'home_team', 'away_team', 'game_strength_state']]

            pbp_with_xg = pbp_with_xg[pbp_with_xg.game_strength_state.isin(all_strength_states)]

            # Filter by game_id if provided
            if game_id:
                # Convert game_id to integer for comparison
                try:
                    game_id_int = int(game_id)
                    print(f"Filtering by game_id (as int): {game_id_int}")
                    pbp_with_xg = pbp_with_xg[pbp_with_xg['game_id'] == game_id_int]
                except ValueError:
                    # If game_id is not a valid integer, try string comparison
                    print(f"Filtering by game_id (as string): {game_id}")
                    pbp_with_xg = pbp_with_xg[pbp_with_xg['game_id'] == game_id]
            
            # Additional filtering by home_team and away_team if provided
            if home_team and away_team:
                # Handle team abbreviation variations
                home_team_normalized = home_team.replace('LAK', 'L.A').replace('NJD', 'N.J').replace('TBL', 'T.B').replace('SJS', 'S.J')
                away_team_normalized = away_team.replace('LAK', 'L.A').replace('NJD', 'N.J').replace('TBL', 'T.B').replace('SJS', 'S.J')
                
                print(f"Filtering by teams - home: {home_team_normalized}, away: {away_team_normalized}")
                
                pbp_with_xg = pbp_with_xg[
                    (pbp_with_xg['home_team'] == home_team_normalized) & 
                    (pbp_with_xg['away_team'] == away_team_normalized)
                ]

            pbp_with_xg = pbp_with_xg.drop_duplicates()
            
            print(f"Filtered result contains {len(pbp_with_xg)} records")
            
            pbp_with_xg_json = json.loads(pbp_with_xg.to_json(orient = 'records'))

            response_data = {'success': True, 'shots': pbp_with_xg_json}
            response = make_response(response_data)
            
            # Differential caching based on game state
            if game_is_finished:
                # Finished games don't change - cache for 5 minutes
                response.headers['Cache-Control'] = 'public, max-age=300, s-maxage=300, stale-while-revalidate=600'
            else:
                # Live games need frequent updates - cache for 10 seconds
                response.headers['Cache-Control'] = 'public, max-age=10, s-maxage=10, stale-while-revalidate=20'
            
            response.headers['ETag'] = f'W/"{game_id or "all"}-{len(pbp_with_xg)}"'
            response.headers['Vary'] = 'Accept-Encoding'
            return response
        else:
            # No games today
            response_data = {'success': True, 'shots': []}
            response = make_response(response_data)
            # Cache for 60 seconds in browser (max-age) and shared caches (s-maxage)
            response.headers['Cache-Control'] = 'public, max-age=60, s-maxage=60, stale-while-revalidate=300'
            response.headers['ETag'] = 'W/"no-games"'
            response.headers['Vary'] = 'Accept-Encoding'
            return response
    except Exception as e:
        print(f"Error in live-games-pbp-python: {e}")
        error_msg = f"ERROR in live-games-pbp-python:\n"
        error_msg += f"Exception: {type(e).__name__}: {str(e)}\n"
        error_msg += f"Traceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        response_data = {'success': True, 'shots': []}
        response = make_response(response_data)
        return response