import json
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from databallpy.load_data.metadata import Metadata
from databallpy.utils.tz_modification import utc_to_local_datetime
from databallpy.utils.utils import MISSING_INT
from databallpy.warnings import DataBallPyWarning

ORTEC_EVENTS_MAP = {
    0: "Unknown",
    1: "Action",
    2: "ActionSuccesfull",
    3: "Passes",
    4: "PassCompleted",
    5: "PassForward",
    6: "PassFowardCompleted",
    7: "PassWide",
    8: "PassWideCompleted",
    9: "PassBackward",
    10: "PassBackwardCompleted",
    11: "PassLongForward",
    12: "PassShort",
    13: "PassFirstInPossession",
    14: "PassForwardFirstInPossession",
    15: "PassBetweenCentralDefenders",
    16: "PassOpponentHalf",
    17: "PassInFinalThird",
    18: "PassInFinalThirdCompleted",
    19: "PassCross",
    20: "PassCrossEarly",
    21: "PassCrossLate",
    22: "PassToOpponentPenaltyBox",
    23: "PassDirect",
    24: "ActionKey",
    25: "CornerKeyAction",
    26: "FreeKickKeyAction",
    27: "ActionAssist",
    28: "GoalAttemptScored",
    29: "GoalWithHead",
    30: "Goalattempt",
    31: "GoalAttemptOnTarget",
    32: "ShotInsideOpponentPenaltyBox",
    33: "ShotInsideOpponentPenaltyBoxScored",
    34: "ShotInsideOpponentPenaltyBoxOnTarget",
    35: "ShotOutsideOpponentPenaltyBox",
    36: "ShotOutsideOpponentPenaltyBoxScored",
    37: "ShotOutsideOpponentPenaltyBoxOnTarget",
    38: "FreeKickOnGoal",
    39: "ShotWithHead",
    40: "PossessionRegainByDuel",
    41: "Interception",
    42: "PossessionRegainInPlayOpponentHalf",
    43: "PossessionLoss",
    44: "PossessionLossByPass",
    45: "PossessionLossByDuel",
    46: "ActionClearance",
    47: "ActionSliding",
    48: "DuelDefendingPartGroundStanding",
    49: "DuelDefendingPartGroundStandingWon",
    50: "DuelDefendingPartAir",
    51: "DuelDefendingPartAirWon",
    52: "DuelDefendingPart",
    53: "DuelWonByDefender",
    54: "DuelAttackingPartGroundStanding",
    55: "DuelAttackingPartGroundStandingWon",
    56: "DuelAttackingPartAir",
    57: "DuelAttackingPartAirWon",
    58: "DuelAttackingPart",
    59: "DuelWonByAttacker",
    60: "DuelDefendingPartOpponentHalf",
    61: "DuelDefendingPartOpponentHalfWon",
    62: "DuelDefendingPartOwnHalf",
    63: "DuelDefendingPartOwnHalfWon",
    64: "Offside",
    65: "Dribble",
    66: "Corner",
    67: "ThrowIn",
    68: "Foul",
    69: "SaveOnGoalAttempt",
    70: "FreeKick",
    71: "KeeperThrowShort",
    72: "KeeperThrowLong",
    73: "PassSwitchOffPlay",
    74: "PassInBox",
    75: "DribbleSuccesfull",
    76: "ActionOpponentBox",
    77: "BlockedShot",
    78: "CornerShort",
    79: "CornerLong",
    80: "Possession",
    81: "PossessionOppHalf",
    82: "ThrowinKeepPossession",
    83: "FoulSuffered",
    84: "Penalty",
    85: "FiftyFiftyDuel",
    87: "TotalXLocation",
    88: "TotalYLocation",
    89: "Grade",
    90: "Speed",
    91: "PossessionTime",
    92: "PassesOwnHalf",
    93: "PassesOwnHalfCompleted",
    94: "PassesOpponentHalfCompleted",
    95: "Keypasses",
    96: "PassCrossCompleted",
    97: "PassCrossHigh",
    98: "PassCrossLow",
    99: "PassCrossToGoalAttempt",
    100: "PassCrossToGoal",
    101: "DirectFreekickScored",
    102: "DirectFreekkickMissed",
    103: "AssistsByFreekick",
    104: "PenaltyScored",
    105: "DefensiveDuelsOwnBox",
    106: "DefensiveDuelsOwnboxWon",
    107: "PossessionRegainInPlay",
    108: "PossessionRegainInPlayOwnHalf",
    109: "InterceptionByGoalkeeper",
    111: "GoalKicks",
    112: "YellowCards",
    113: "RedCards",
    114: "DirectRedCards",
    115: "FoulsOwnHalf",
    116: "OwnGoals",
    117: "DefensiveBlocks",
    118: "PassLongForwardCompleted",
    119: "DuelsTotal",
    120: "DuelsTotalWon",
}

ORTEC_DATABALLPY_MAP = {}
for ortec_event in ORTEC_EVENTS_MAP.values():
    if (
        "pass" in ortec_event.lower()
        or "corner" in ortec_event.lower()
        or "throw" in ortec_event.lower()
        or "cross" in ortec_event.lower()
    ):
        ORTEC_DATABALLPY_MAP[ortec_event] = "pass"
    elif "goal" in ortec_event.lower() or "shot" in ortec_event.lower():
        ORTEC_DATABALLPY_MAP[ortec_event] = "shot"
    elif (
        "dribble" in ortec_event.lower()
        or "duelattackingpartground" in ortec_event.lower()
    ):
        ORTEC_DATABALLPY_MAP[ortec_event] = "dribble"


ORTEC_COMPLETED_PROPERTY_ID = 8


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
    if event_data_loc is not None:
        warnings.warn(
            "event_data_loc is not used in this function. It is only used in the "
            "load_tracking_data function",
            DataBallPyWarning,
        )
        # if not isinstance(event_data_loc, (str, type(None))):
        #     raise TypeError("event_data_loc should be a string")
    if not isinstance(metadata_loc, str):
        raise TypeError("metadata_loc should be a string")
        # if event_data_loc is not None and not event_data_loc.endswith(".json"):
        #     raise ValueError("event_data_loc should be a .json file")

    metadata = _load_metadata(metadata_loc)
    # metadata, possessions = _update_metadata(metadata, event_data_loc)
    # event_data = _load_ortec_event_data(event_data_loc=event_data_loc)
    # home_mask = event_data["team_id"] == metadata.home_team_id
    # away_mask = event_data["team_id"] == metadata.away_team_id
    # event_data.loc[home_mask, "player_name"] = event_data.loc[
    #     home_mask, "player_id"
    # ].map(metadata.home_players.set_index("id")["full_name"])
    # event_data.loc[away_mask, "player_name"] = event_data.loc[
    #     away_mask, "player_id"
    # ].map(metadata.away_players.set_index("id")["full_name"])
    return None, metadata


def _update_metadata(
    metadata: Metadata, event_data_loc: str
) -> Tuple[Metadata, pd.DataFrame]:
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
        players_df = (
            metadata.home_players
            if player_dict["Person"] in metadata.home_players["id"].to_list()
            else metadata.away_players
        )
        shirt_num = player_dict["ShirtNumber"]
        if shirt_num in players_df.shirt_num.values:
            players_df.loc[
                players_df.shirt_num == shirt_num, "minutes_played"
            ] = player_dict["MinutesPlayed"]

    metadata.home_score = data["HomeScore"]
    metadata.away_score = data["AwayScore"]

    # optain possession information
    temp_possessions = pd.DataFrame(data["PossessionMoments"])
    possessions_df = pd.DataFrame(index=range(len(temp_possessions)))
    possessions_df["start"] = temp_possessions["StartTime"] * 0.002
    possessions_df["end"] = temp_possessions["EndTime"] * 0.002
    possessions_df["team"] = temp_possessions["SelectionInPossession"].map(
        {metadata.home_team_id: "home", metadata.away_team_id: "away"}
    )

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
        "home_score": [],
        "away_score": [],
        "to_player_id": [],
        "to_team_id": [],
        "end_x": [],
        "end_y": [],
    }
    to_add_minutes = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}

    for event in data["Events"]:
        result_dict["event_id"].append(event["Id"])
        result_dict["period_id"].append(event["Phase"])
        milliseconds = event["Time"]
        seconds = milliseconds / 1000
        result_dict["minutes"].append(seconds // 60 + to_add_minutes[event["Phase"]])
        result_dict["seconds"].append(seconds % 60)
        result_dict["datetime"].append(pd.to_datetime(event["DateTime"], utc=True))
        result_dict["home_score"].append(event["Score"][0])
        result_dict["away_score"].append(event["Score"][1])
        annot = event["Annotations"][0]
        result_dict["player_id"].append(
            annot["PersonId"] if "PersonId" in annot.keys() else MISSING_INT
        )
        result_dict["team_id"].append(
            annot["SelectionEditionId"]
            if "SelectionEditionId" in annot.keys()
            else MISSING_INT
        )
        result_dict["type_id"].append(
            annot["Label"]
        ) if "Label" in annot.keys() else MISSING_INT
        event_name = (
            ORTEC_EVENTS_MAP[annot["Label"]] if "Label" in annot.keys() else "Unknown"
        )
        result_dict["ortec_event"].append(event_name)
        result_dict["start_x"].append(
            annot["LocationX"] if "LocationX" in annot.keys() else np.nan
        )
        result_dict["start_y"].append(
            annot["LocationY"] if "LocationY" in annot.keys() else np.nan
        )
        result_dict["to_player_id"].append(
            annot["NextPersonId"] if "NextPersonId" in annot.keys() else MISSING_INT
        )
        result_dict["to_team_id"].append(
            annot["NextSelectionEdition"]
            if "NextSelectionEdition" in annot.keys()
            else MISSING_INT
        )
        result_dict["end_x"].append(
            annot["NextLocationX"] if "NextLocationX" in annot.keys() else np.nan
        )
        result_dict["end_y"].append(
            annot["NextLocationY"] if "NextLocationY" in annot.keys() else np.nan
        )
        if len(annot["Properties"]) > 0:
            result_dict["outcome"].append(
                [0, 1][ORTEC_COMPLETED_PROPERTY_ID in annot["Properties"]]
            )
        else:
            result_dict["outcome"].append(MISSING_INT)

    result_dict["databallpy_event"] = [None] * len(result_dict["event_id"])

    df = pd.DataFrame(result_dict)
    df["databallpy_event"] = df["ortec_event"].map(ORTEC_DATABALLPY_MAP)

    return df


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
