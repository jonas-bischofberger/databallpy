import json
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.utils.tz_modification import utc_to_local_datetime
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning

import json


def load_ortec_event_data(
    event_data_loc: str, metadata_loc: str
) -> Tuple[pd.DataFrame, Metadata]:
    """Function that loads the event data from Ortec

    Args:
        event_data_loc (str): location of the event data
        metadata_loc (str): location of the metadata

    Returns:
        Tuple[pd.DataFrame, Metadata]: event data and metadata of the match
    """
    if not isinstance(event_data_loc, str):
        raise TypeError("event_data_loc should be a string")
    if not isinstance(metadata_loc, str):
        raise TypeError("metadata_loc should be a string")
    if not event_data_loc.endswith(".json"):
        raise ValueError("event_data_loc should be a .json file")

    metadata = _load_metadata(metadata_loc)
    metadata, possessions = _update_metadata(metadata, event_data_loc)
    event_data = _load_ortec_event_data(event_data_loc=event_data_loc)
    return event_data, metadata, databallpy_events

def _update_metadata(metadata: Metadata, event_data_loc: str) -> Tuple[Metadata, pd.DataFrame]:
    """Function that updates the metadata with the event data. The function 
    updates the player information and the score of the match. It also obtains
    the possession information to enrich potential tracking data.

    Args:
        metadata (Metadata): metadata of the match
        event_data_loc (str): location of the event data

    Returns:
        Tuple[Metadata, pd.DataFrame]: updated metadata and possession information
    """
    with open(event_data_loc, "r") as f:
        data = json.load(f)

     # first update the player information in the metadata
    for player_dict in data["LineUp"]:
        players_df = metadata.home_players if player_dict["Person"] in metadata.home_players["id"].to_list() else metadata.away_players
        shirt_num = player_dict["ShirtNumber"]
        if shirt_num in players_df.shirt_num.values:
            players_df.loc[players_df.shirt_num == shirt_num, "minutes_played"] = player_dict["MinutesPlayed"]

    metadata.home_score = data["HomeScore"]
    metadata.away_score = data["AwayScore"]

    # optain possession information
    temp_possessions = pd.DataFrame(data["PossessionMoments"])
    possessions_df = pd.DataFrame(index=range(len(temp_possessions)))
    possessions_df["start"] = temp_possessions["StartTime"] * 0.002
    possessions_df["end"] = temp_possessions["EndTime"] * 0.002
    possessions_df["team"] = temp_possessions["SelectionInPossession"].map({metadata.home_team_id: "home", metadata.away_team_id: "away"})

    return metadata, possessions_df

def _load_ortec_event_data(event_data_loc: str) -> pd.DataFrame:
    with open(event_data_loc, "r") as f:
        data = json.load(f)
    
    result_dict = {
        "event_id": [],
        "type_id": [],
        "databallpy_event": [],
        "period_id": [],
        "minutes": [],
        "seconds": [],
        "player_id": [],
        "team_id": [],
        "outcome": [],
        "start_x": [],
        "start_y": [],
        "datetime": [],
        "ortec_event": [],
    }
    for event in data["Events"]:

    
   

        import pdb; pdb.set_trace()


def _get_player_info(info_players: list) -> pd.DataFrame:
    """Function that gets the information of the players

    Args:
        info_players (list): list with all players

    Returns:
        pd.DataFrame: dataframe with all player info
    """
    team_info = {
        "id": [],
        "full_name": [],
        "position": [],
        "starter": [],
        "shirt_num": [],
    }

    for player in info_players:
        team_info["full_name"].append(str(player["DisplayName"]))
        team_info["id"].append(int(player["Id"]))
        team_info["position"].append(str(player["Role"]))
        team_info["shirt_num"].append(int(player["ShirtNumber"]))
        if str(player["Role"]) == "bench":
            team_info["starter"].append(False)
        else:
            team_info["starter"].append(True)

    return (
        pd.DataFrame(team_info)
        .sort_values("starter", ascending=False)
        .reset_index(drop=True)
    )


def _load_metadata(metadata_loc: str) -> pd.DataFrame:
    """Function that loads metadata from .json file

    Args:
        metadata_loc (str): location of the .json file

    Returns:
        pd.DataFrame: metadata of the match
    """
    with open(metadata_loc, "r") as f:
        data = f.read()
    metadata_json = json.loads(data)

    periods = {
        "period": [1, 2, 3, 4, 5],
        "start_datetime_ed": [pd.to_datetime("NaT")] * 5,
        "end_datetime_ed": [pd.to_datetime("NaT")] * 5,
    }
    country = metadata_json["Competition"]["Name"].split()[0]
    datetime = pd.to_datetime(metadata_json["DateTime"], utc=True)
    start_datetime = utc_to_local_datetime(datetime, country)
    periods["start_datetime_ed"][0] = start_datetime
    periods["end_datetime_ed"][0] = start_datetime + pd.Timedelta(45, "minutes")
    periods["start_datetime_ed"][1] = start_datetime + pd.Timedelta(60, "minutes")
    periods["end_datetime_ed"][1] = start_datetime + pd.Timedelta(105, "minutes")

    info_home_players = metadata_json["HomeTeam"]["Persons"]
    info_away_players = metadata_json["AwayTeam"]["Persons"]
    home_players = _get_player_info(info_home_players)
    away_players = _get_player_info(info_away_players)

    home_formation = _get_formation(home_players)
    away_formation = _get_formation(away_players)

    metadata = Metadata(
        match_id=metadata_json["Id"],
        pitch_dimensions=[np.nan, np.nan],
        periods_frames=pd.DataFrame(periods),
        frame_rate=MISSING_INT,
        home_team_id=metadata_json["HomeTeam"]["Id"],
        home_team_name=str(metadata_json["HomeTeam"]["DisplayName"]),
        home_players=home_players,
        home_score=MISSING_INT,
        home_formation=home_formation,
        away_team_id=metadata_json["AwayTeam"]["Id"],
        away_team_name=str(metadata_json["AwayTeam"]["DisplayName"]),
        away_players=away_players,
        away_score=MISSING_INT,
        away_formation=away_formation,
        country=country,
    )
    return metadata


def _get_formation(players_info: pd.DataFrame) -> str:
    """Function that gets the formation of the team

    Args:
        players_info (pd.DataFrame): dataframe with all player info

    Returns:
        str: formation of the team
    """
    gk = 0
    defenders = 0
    midfielders = 0
    attackers = 0

    for position in players_info.loc[players_info["position"] != "bench", "position"]:
        if "keeper" in position.lower():
            gk += 1
        if "back" in position.lower():
            defenders += 1
        if "midfield" in position.lower():
            midfielders += 1
        if "forward" in position.lower():
            attackers += 1
    return f"{gk}{defenders}{midfielders}{attackers}"
