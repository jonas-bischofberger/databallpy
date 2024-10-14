import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import pandas as pd

import sklearn.metrics
import numpy as np

import assets.test_data
import streamlit as st

import dangerous_accessible_space.core

DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0 = 3
DEFAULT_FALLBACK_V0 = 10
DEFAULT_USE_POSS_FOR_XC = False  # False
DEFAULT_USE_FIXED_V0_FOR_XC = True
DEFAULT_V0_MAX_FOR_XC = 15.108273248071049
DEFAULT_V0_MIN_FOR_XC = 4.835618861117393
DEFAULT_N_V0_FOR_XC = 6

DEFAULT_N_ANGLES_FOR_DAS = 60
DEFAULT_PHI_OFFSET = 0
DEFAULT_N_V0_FOR_DAS = 20
DEFAULT_V0_MIN_FOR_DAS = 3#0.01
DEFAULT_V0_MAX_FOR_DAS = 30


def get_matrix_coordinates(
    df_tracking, frame_col="frame_id", player_col="player_id", ball_player_id="ball", team_col="team_id",
    controlling_team_col="ball_possession", x_col="x", y_col="y", vx_col="vx", vy_col="vy"
):
    """
    Convert tracking data from a DataFrame to numpy matrices as used internally to compute the passing model.

    >>> assets.test_data.df_tracking
         frame_id player_id  team_id    x     y   vx    vy
    0           0         A      0.0 -0.1  0.00  0.1  0.05
    1           1         A      0.0  0.0  0.05  0.1  0.05
    2           2         A      0.0  0.1  0.10  0.1  0.05
    3           3         A      0.0  0.2  0.15  0.1  0.05
    4           4         A      0.0  0.3  0.20  0.1  0.05
    ..        ...       ...      ...  ...   ...  ...   ...
    114        15      ball      NaN  1.5  0.00  0.1  0.00
    115        16      ball      NaN  1.6  0.00  0.1  0.00
    116        17      ball      NaN  1.7  0.00  0.1  0.00
    117        18      ball      NaN  1.8  0.00  0.1  0.00
    118        19      ball      NaN  1.9  0.00  0.1  0.00
    <BLANKLINE>
    [119 rows x 7 columns]
    >>> PLAYER_POS, BALL_POS, player_list, team_indices = _get_matrix_coordinates(assets.test_data.df_tracking)
    >>> PLAYER_POS.shape, BALL_POS.shape, players.shape, player_teams.shape
    ((20, 5, 4), (20, 4), (5,), (5,))
    """
    df_tracking = df_tracking.sort_values(by=[frame_col, team_col])

    i_player = df_tracking[player_col] != ball_player_id

    df_players = df_tracking.loc[i_player].pivot(
        index=frame_col, columns=player_col, values=[x_col, y_col, vx_col, vy_col]
    )
    F = df_players.shape[0]  # number of frames
    C = 4  # number of coordinates per player
    P = df_tracking.loc[i_player, player_col].nunique()  # number of players

    dfp = df_players.stack(level=1, dropna=False)
    PLAYER_POS = dfp.values.reshape(F, P, C)
    frame_to_idx = {frame: i for i, frame in enumerate(df_players.index)}

    players = df_players.columns.get_level_values(1).unique()  # P
    player2team = df_tracking.loc[i_player, [player_col, team_col]].drop_duplicates().set_index(player_col)[team_col]
    player_teams = player2team.loc[players].values

    df_ball = df_tracking.loc[~i_player].set_index(frame_col)[[x_col, y_col, vx_col, vy_col]]
    BALL_POS = df_ball.values  # F x C

    controlling_teams = df_tracking.groupby(frame_col)[controlling_team_col].first().values

    F = PLAYER_POS.shape[0]
    assert F == BALL_POS.shape[0]
    assert F == controlling_teams.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but passer_team shape is {controlling_teams.shape}"
    P = PLAYER_POS.shape[1]
    assert P == player_teams.shape[0]
    assert P == players.shape[0]
    assert PLAYER_POS.shape[2] >= 4  # >= or = ?
    assert BALL_POS.shape[1] >= 2  # ...

    return PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx


def per_object_frameify_tracking_data(
        df_tracking, frame_col, x_cols, y_cols, vx_cols, vy_cols, players, player_to_team, new_x_col="x", new_y_col="y",
        new_vx_col="vx", new_vy_col="vy", new_player_col="player_id", new_team_col="team_id", v_cols=None, new_v_col="v",
):
    """ Converts tracking data with '1 row per frame' into '1 row per frame + player' format """
    dfs_player = []
    for player_nr, player in enumerate(players):
        coordinate_cols = [x_cols[player_nr], y_cols[player_nr], vx_cols[player_nr], vy_cols[player_nr]]
        coordinate_mapping = {x_cols[player_nr]: new_x_col, y_cols[player_nr]: new_y_col, vx_cols[player_nr]: new_vx_col, vy_cols[player_nr]: new_vy_col}
        if v_cols is not None:
            coordinate_cols.append(v_cols[player_nr])
            coordinate_mapping[v_cols[player_nr]] = new_v_col
        df_player = df_tracking[[frame_col] + coordinate_cols]
        df_player = df_player.rename(columns=coordinate_mapping)
        df_player[new_player_col] = player
        df_player[new_team_col] = player_to_team.get(player, None)
        dfs_player.append(df_player)

    df_player = pd.concat(dfs_player, axis=0)

    all_coordinate_columns = x_cols + y_cols + vx_cols + vy_cols
    if v_cols is not None:
        all_coordinate_columns += v_cols

    remaining_cols = [col for col in df_tracking.columns if col not in [frame_col] + all_coordinate_columns]

    return df_player.merge(df_tracking[[frame_col] + remaining_cols], on=frame_col, how="left")


def _get_unused_column_name(df, prefix):
    i = 1
    new_column_name = prefix
    while new_column_name in df.columns:
        new_column_name = f"{prefix}_{i}"
        i += 1
    return new_column_name


def get_pass_velocity(
    df_passes, df_tracking_ball, event_frame_col="frame_id", tracking_frame_col="frame_id",
    n_frames_after_pass_for_v0=DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0, fallback_v0=DEFAULT_FALLBACK_V0, tracking_vx_col="vx",
    tracking_vy_col="vy", tracking_v_col=None
):
    """
    Add initial velocity to passes according to the first N frames of ball tracking data after the pass

    >>> df_passes = assets.test_data.df_passes
    >>> df_passes["v0"] = get_pass_velocity(df_passes, assets.test_data.df_tracking[assets.test_data.df_tracking["player_id"] == "ball"])
    >>> df_passes
       frame_id player_id receiver_id  ...  y_target  pass_outcome   v0
    0         0         A           B  ...        11    successful  0.1
    1         6         B           X  ...        30        failed  0.1
    2        14         C           Y  ...        -1        failed  0.1
    <BLANKLINE>
    [3 rows x 10 columns]
    """
    df_passes = df_passes.copy()
    df_tracking_ball = df_tracking_ball.copy()
    pass_nr_col = _get_unused_column_name(df_passes, "pass_nr")
    frame_end_col = _get_unused_column_name(df_passes, "frame_end")
    ball_velocity_col = _get_unused_column_name(df_tracking_ball, "ball_velocity")

    df_passes[pass_nr_col] = df_passes.index
    df_tracking_ball = df_tracking_ball.merge(df_passes[[event_frame_col, pass_nr_col]], left_on=tracking_frame_col, right_on=event_frame_col, how="left")

    fr_max = df_tracking_ball[tracking_frame_col].max()
    df_passes[frame_end_col] = np.minimum(df_passes[event_frame_col] + n_frames_after_pass_for_v0 - 1, fr_max)

    all_valid_frame_list = np.concatenate([np.arange(start, end + 1) for start, end in zip(df_passes[event_frame_col], df_passes[frame_end_col])])

    df_tracking_ball_v0 = df_tracking_ball[df_tracking_ball[tracking_frame_col].isin(all_valid_frame_list)]
    # df_tracking_ball_v0 = df_tracking_ball[df_tracking_ball[frame_col].apply(lambda x: any(start <= x <= end for start, end in zip(df_passes[frame_col], df_passes["frame_end"])))]
    df_tracking_ball_v0[pass_nr_col] = df_tracking_ball_v0[pass_nr_col].ffill()
    if tracking_v_col is not None:
        df_tracking_ball_v0[ball_velocity_col] = df_tracking_ball_v0[tracking_v_col]
    else:
        df_tracking_ball_v0[ball_velocity_col] = np.sqrt(df_tracking_ball_v0[tracking_vx_col] ** 2 + df_tracking_ball_v0[tracking_vy_col] ** 2)

    dfg_v0 = df_tracking_ball_v0.groupby(pass_nr_col)[ball_velocity_col].mean()

    v0 = df_passes[pass_nr_col].map(dfg_v0)
    v0 = v0.fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames
    return v0


def get_expected_pass_completion(
    df_passes, df_tracking, tracking_frame_col="frame_id", event_frame_col="frame_id",
    tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball",
    n_frames_after_pass_for_v0=5, fallback_v0=10, tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx",
    tracking_vy_col="vy", tracking_v_col=None, event_start_x_col="x", event_start_y_col="y",
    event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="",
    outcome_col="success",

    # xC Parameters
    exclude_passer=True,
    use_poss=DEFAULT_USE_POSS_FOR_XC,
    use_fixed_v0=DEFAULT_USE_FIXED_V0_FOR_XC,
    v0_min=DEFAULT_V0_MIN_FOR_XC,
    v0_max=DEFAULT_V0_MAX_FOR_XC,
    n_v0=DEFAULT_N_V0_FOR_XC,

    # Core model parameters
    pass_start_location_offset=dangerous_accessible_space.DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=dangerous_accessible_space.DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=dangerous_accessible_space.DEFAULT_RADIAL_GRIDSIZE,
    # seconds_to_intercept=DEFAULT_SECONDS_TO_INTERCEPT,
    b0=dangerous_accessible_space.DEFAULT_B0,
    b1=dangerous_accessible_space.DEFAULT_B1,
    player_velocity=dangerous_accessible_space.DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=dangerous_accessible_space.DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=dangerous_accessible_space.DEFAULT_USE_MAX,
    v_max=dangerous_accessible_space.DEFAULT_V_MAX,
    a_max=dangerous_accessible_space.DEFAULT_A_MAX,
    inertial_seconds=dangerous_accessible_space.DEFAULT_INERTIAL_SECONDS,
    tol_distance=dangerous_accessible_space.DEFAULT_TOL_DISTANCE,
    use_approx_two_point=dangerous_accessible_space.DEFAULT_USE_APPROX_TWO_POINT,
):
    df_passes = df_passes.sort_values(event_frame_col).copy()

    # 1. Extract player and ball positions at passes
    assert set(df_passes[event_frame_col]).issubset(set(df_tracking[tracking_frame_col]))
    i_pass_in_tracking = df_tracking[tracking_frame_col].isin(df_passes[event_frame_col])
    PLAYER_POS, BALL_POS, players, player_teams, _, frame_to_idx = dangerous_accessible_space.get_matrix_coordinates(
        df_tracking.loc[i_pass_in_tracking], frame_col=tracking_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col,
    )

    # 2. Add v0 to passes
    v0_col = _get_unused_column_name(df_passes, "v0")
    df_passes[v0_col] = get_pass_velocity(
        df_passes, df_tracking[df_tracking[tracking_player_col] == ball_tracking_player_id],
        event_frame_col=event_frame_col, tracking_frame_col=tracking_frame_col,
        n_frames_after_pass_for_v0=n_frames_after_pass_for_v0, fallback_v0=fallback_v0, tracking_vx_col=tracking_vx_col,
        tracking_vy_col=tracking_vy_col, tracking_v_col=tracking_v_col
    )
    # v0_grid = df_passes[v0_col].values[:, np.newaxis]  # F x V0
    if use_fixed_v0:
        v0_grid = np.linspace(start=v0_min, stop=v0_max, num=round(n_v0))[np.newaxis, :].repeat(df_passes.shape[0], axis=0)  # F x V0
    else:
        v0_grid = df_passes[v0_col].values[:, np.newaxis]  # F x V0=1, only simulate actual passing speed

    # 3. Add angle to passes
    phi_col = _get_unused_column_name(df_passes, "phi")
    df_passes[phi_col] = np.arctan2(df_passes[event_end_y_col] - df_passes[event_start_y_col], df_passes[event_end_x_col] - df_passes[event_start_x_col])
    phi_grid = df_passes[phi_col].values[:, np.newaxis]  # F x PHI

    # 4. Extract player team info
    passer_teams = df_passes[event_team_col].values  # F
    player_teams = np.array(player_teams)  # P
    if exclude_passer:
        passers_to_exclude = df_passes[event_player_col].values  # F
    else:
        passers_to_exclude = None

    # st.write("df_passes", df_passes.shape)
    # st.write(df_passes)

    # 5. Simulate passes to get expected completion
    simulation_result = dangerous_accessible_space.simulate_passes(
        # xC parameters
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players,
        passers_to_exclude=passers_to_exclude,

        # Core model parameters
        pass_start_location_offset=pass_start_location_offset,
        time_offset_ball=time_offset_ball,
        radial_gridsize=radial_gridsize,
        b0=b0,
        b1=b1,
        player_velocity=player_velocity,
        keep_inertial_velocity=keep_inertial_velocity,
        use_max=use_max,
        v_max=v_max,
        a_max=a_max,
        inertial_seconds=inertial_seconds,
        tol_distance=tol_distance,
        use_approx_two_point=use_approx_two_point,
    )
    if use_poss:
        xc = simulation_result.poss_cum_att[:, 0, -1]  # F x PHI x T ---> F
    else:
        xc = simulation_result.prob_cum_att[:, 0, -1]  # F x PHI x T ---> F

    outcomes = df_passes[outcome_col].values

    brier = sklearn.metrics.brier_score_loss(outcomes, xc)
    logloss = sklearn.metrics.log_loss(outcomes, xc, labels=[0, 1])
    average_completion = np.mean(outcomes)
    brier_baseline = np.mean((outcomes - average_completion)**2)
    brier_skill_score = 1 - brier / brier_baseline

    idx = df_passes[event_frame_col].map(frame_to_idx)

    return xc, idx, simulation_result


def get_dangerous_accessible_space(
    # Data
    df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
    ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy",

    # Options
    apply_pitch_mask_to_raw_result=False,

    # Parameters
    n_angles=DEFAULT_N_ANGLES_FOR_DAS,
    phi_offset=DEFAULT_PHI_OFFSET,
    n_v0=DEFAULT_N_V0_FOR_DAS,
    v0_min=DEFAULT_V0_MIN_FOR_DAS,
    v0_max=DEFAULT_V0_MAX_FOR_DAS,
):
    PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx = dangerous_accessible_space.get_matrix_coordinates(
        df_tracking, frame_col=tracking_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col,
    )
    F = PLAYER_POS.shape[0]

    phi_grid = np.tile(np.linspace(phi_offset, 2*np.pi+phi_offset, n_angles, endpoint=False), (F, 1))  # F x PHI

    v0_grid = np.tile(np.linspace(v0_min, v0_max, n_v0), (F, 1))  # F x V0

    simulation_result = dangerous_accessible_space.simulate_passes_chunked(
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, controlling_teams, player_teams, players, passers_to_exclude=None,
    )
    if apply_pitch_mask_to_raw_result:
        simulation_result = dangerous_accessible_space.mask_out_of_play(simulation_result)

    accessible_space = dangerous_accessible_space.aggregate_surface_area(simulation_result)  # F
    fr2AS = pd.Series(accessible_space, index=df_tracking[tracking_frame_col].unique())
    # df_tracking["AS"] = df_tracking[tracking_frame_col].map(fr2AS)

    # st.write("AS", accessible_space.shape, accessible_space)
    # st.write("df_tracking", df_tracking.shape)
    # st.write(df_tracking)
    # dangerous_accessible_space = aggregate_surface_area(simulation_result_das)
    # st.write("DAS", dangerous_accessible_space)
    # df_passes["DAS"] = dangerous_accessible_space

    accessible_space_series = df_tracking[tracking_frame_col].map(fr2AS)
    idx = df_tracking[tracking_frame_col].map(frame_to_idx)

    return accessible_space_series, idx, simulation_result

    # return simulation_result
