import warnings

import pandas as pd
import numpy as np

from databallpy.utils.logging import create_logger

LOGGER = create_logger(__name__)

def _parse_intervals(intervals):
    if not all(isinstance(element, (int, float)) for element in intervals):
        raise TypeError("All elements in the tuple must be integers or floats")
    
    if len(intervals) % 2 != 0:
        raise ValueError("Intervals must contain an even number of elements.")
    
    return [(min(intervals[i], intervals[i+1]), max(intervals[i], intervals[i+1])) for i in range(0, len(intervals), 2)]

def get_covered_distance(
    tracking_data: pd.DataFrame,
    player_ids: list[str],
    framerate: int,
    velocity_intervals: tuple[float, ...] | None = None,
    acceleration_intervals: tuple[float, ...] | None = None
    ) -> dict:
    """Function that calculates the distance covered in specified velocity and acceleration intervals.

    Args:
        tracking_data:      tracking data
        player_ids:         list containing players. Example = ['home_8', 'away_3', 'home_16']
        framerate:          sample frequency in Hertz
        velocity_intervals: tuple that contains the velocity interval(s). For example [0 3] & [4 8]: (0,3,8,4)
        acc_intervals:      tuple that contains the acceleration interval(s). For example [-3 0] & [3 5]: (-3,0,3,5)      

    Returns:
        dict: a dictionary with for every player the total distance covered and optionally the 
        distance covered within the given velocity and/or acceleration threshold(s)
    
    Notes: 
        The function requires the velocity for every player calculated with the add_velocity function. 
        The acceleration for every player depends on the presence of acceleration intervals in the input
    """

    try:
        # Check input types
        # tracking_data
        if not isinstance(tracking_data, pd.DataFrame):
            raise TypeError(f"tracking data must be a pandas DataFrame, not a {type(tracking_data).__name__}")
        
        # player_ids
        if not isinstance(player_ids, list):
            raise TypeError(f"player_ids must be a list, not a {type(player_ids).__name__}")

        if not all(isinstance(player, str) for player in player_ids):
            raise TypeError("All elements in player_ids must be strings")
        
        # framerate
        if not isinstance(framerate, int):
            raise TypeError(f"framerate must be a int, not a {type(framerate).__name__}")

        # check velocity
        for player_id in player_ids:
            if (
                player_id + "_vx" not in tracking_data.columns
                or player_id + "_vy" not in tracking_data.columns
                or player_id + "_velocity" not in tracking_data.columns
            ):
                raise ValueError(
                    f"Velocity was not found for {player_id} in the DataFrame. "
                    " Please calculate velocity first using add_velocity() function."
                )

        # initialize dictionary covered distance
        players = {player_id: {'total_distance': 0} for player_id in player_ids}

        # Calculate total distance
        tracking_data_velocity = pd.concat([tracking_data[player_id + '_velocity'] for player_id in player_ids], axis=1)
        total_distance = tracking_data_velocity.apply(lambda player: np.sum(player / framerate))
        
        for i, player_id in enumerate(player_ids):
            players[player_id]['total_distance'] = total_distance.iloc[i]

        # Calculate velocity distance
        if velocity_intervals is not None:
            velocity_intervals = _parse_intervals(velocity_intervals)
            for player_id in player_ids:
                # initialize dictionary for velocity intervals
                players[player_id]['total_distance_velocity'] = []
                for min_vel, max_vel in velocity_intervals: 
                    mask = (tracking_data_velocity[player_id + '_velocity'] >= min_vel) & (tracking_data_velocity[player_id + '_velocity'] <= max_vel)
                    filtered_velocities = tracking_data_velocity[player_id + '_velocity'][mask]
                    distance_covered = np.sum(filtered_velocities / framerate)
                    players[player_id]['total_distance_velocity'].append(((min_vel, max_vel), distance_covered))

        # Calculate acceleration distance
        if acceleration_intervals is not None:
            # check acceleration
            for player_id in player_ids:
                if (
                player_id + "_ax" not in tracking_data.columns
                or player_id + "_ay" not in tracking_data.columns
                or player_id + "_acceleration" not in tracking_data.columns
            ):
                    raise ValueError(
                        f"Acceleration was not found for {player_id} in the DataFrame. "
                        " Please calculate acceleration first using add_acceleration() function."
                    )
                else: tracking_data_acceleration = pd.concat([tracking_data[player_id + '_acceleration'] for player_id in player_ids], axis=1)      
            acceleration_intervals = _parse_intervals(acceleration_intervals)
            for player_id in player_ids:
                # initialize dictionary for velocity intervals
                players[player_id]['total_distance_acceleration'] = []
                for min_vel, max_vel in acceleration_intervals: 
                    mask = (tracking_data_acceleration[player_id + '_acceleration'] >= min_vel) & (tracking_data_acceleration[player_id + '_acceleration'] <= max_vel)
                    filtered_acceleration = tracking_data_velocity[player_id + '_velocity'][mask]
                    distance_covered = np.sum(filtered_acceleration / framerate)
                    players[player_id]['total_distance_acceleration'].append(((min_vel, max_vel), distance_covered))

    except Exception as e:
        LOGGER.exception(f"Found unexpected exception in get_covered_distance(): \n{e}")
        raise e
    
    return players
