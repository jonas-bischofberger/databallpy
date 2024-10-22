import numpy as np
import pandas as pd
import colorsys
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.colors

from .assets import test_data
from .core import _DEFAULT_PASS_START_LOCATION_OFFSET, _DEFAULT_B0, _DEFAULT_TIME_OFFSET_BALL, _DEFAULT_A_MAX, \
    _DEFAULT_USE_MAX, _DEFAULT_USE_APPROX_TWO_POINT, _DEFAULT_B1, _DEFAULT_PLAYER_VELOCITY, _DEFAULT_V_MAX, \
    _DEFAULT_KEEP_INERTIAL_VELOCITY, _DEFAULT_INERTIAL_SECONDS, _DEFAULT_TOL_DISTANCE, _DEFAULT_RADIAL_GRIDSIZE, \
    simulate_passes_chunked, simulate_passes, mask_out_of_play, aggregate_surface_area

_DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0 = 3
_DEFAULT_FALLBACK_V0 = 10
_DEFAULT_USE_POSS_FOR_XC = False
_DEFAULT_USE_FIXED_V0_FOR_XC = True
_DEFAULT_V0_MAX_FOR_XC = 15.108273248071049
_DEFAULT_V0_MIN_FOR_XC = 4.835618861117393
_DEFAULT_N_V0_FOR_XC = 6

_DEFAULT_N_ANGLES_FOR_DAS = 60
_DEFAULT_PHI_OFFSET = 0
_DEFAULT_N_V0_FOR_DAS = 20
_DEFAULT_V0_MIN_FOR_DAS = 3
_DEFAULT_V0_MAX_FOR_DAS = 30


def get_matrix_coordinates(
    df_tracking, frame_col="frame_id", player_col="player_id", ball_player_id="ball", team_col="team_id",
    controlling_team_col="ball_possession", x_col="x", y_col="y", vx_col="vx", vy_col="vy"
):
    """
    Convert tracking data from a DataFrame to numpy matrices as used internally to compute the passing model.

    >>> test_data.df_tracking
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
    >>> PLAYER_POS, BALL_POS, player_list, team_indices = get_matrix_coordinates(test_data.df_tracking)
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
    n_frames_after_pass_for_v0=_DEFAULT_N_FRAMES_AFTER_PASS_FOR_V0, fallback_v0=_DEFAULT_FALLBACK_V0, tracking_vx_col="vx",
    tracking_vy_col="vy", tracking_v_col=None
):
    """
    Add initial velocity to passes according to the first N frames of ball tracking data after the pass

    >>> df_passes = test_data.df_passes
    >>> df_passes["v0"] = get_pass_velocity(df_passes, test_data.df_tracking[test_data.df_tracking["player_id"] == "ball"])
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
    use_event_ball_position=False,
    chunk_size=200,

    # xC Parameters
    exclude_passer=True,
    use_poss=_DEFAULT_USE_POSS_FOR_XC,
    use_fixed_v0=_DEFAULT_USE_FIXED_V0_FOR_XC,
    v0_min=_DEFAULT_V0_MIN_FOR_XC,
    v0_max=_DEFAULT_V0_MAX_FOR_XC,
    n_v0=_DEFAULT_N_V0_FOR_XC,

    # Core model parameters
    pass_start_location_offset=_DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=_DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=_DEFAULT_RADIAL_GRIDSIZE,
    b0=_DEFAULT_B0,
    b1=_DEFAULT_B1,
    player_velocity=_DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=_DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=_DEFAULT_USE_MAX,
    v_max=_DEFAULT_V_MAX,
    a_max=_DEFAULT_A_MAX,
    inertial_seconds=_DEFAULT_INERTIAL_SECONDS,
    tol_distance=_DEFAULT_TOL_DISTANCE,
    use_approx_two_point=_DEFAULT_USE_APPROX_TWO_POINT,
):
    df_tracking = df_tracking.copy()

    # 1. Extract player and ball positions at passes
    assert set(df_passes[event_frame_col]).issubset(set(df_tracking[tracking_frame_col]))

    unique_frame_col = _get_unused_column_name(df_passes, "unique_frame")
    df_passes[unique_frame_col] = np.arange(df_passes.shape[0])

    df_tracking_passes = df_passes[[event_frame_col, unique_frame_col]].merge(df_tracking, left_on=event_frame_col, right_on=tracking_frame_col, how="left")
    if use_event_ball_position:
        df_tracking_passes = df_tracking_passes.set_index(unique_frame_col)
        df_passes_copy = df_passes.copy().set_index(unique_frame_col)
        df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_x_col] = df_passes_copy[event_start_x_col]
        df_tracking_passes.loc[df_tracking_passes[tracking_player_col] == ball_tracking_player_id, tracking_y_col] = df_passes_copy[event_start_y_col]
        df_tracking_passes = df_tracking_passes.reset_index()

    PLAYER_POS, BALL_POS, players, player_teams, _, frame_to_idx = get_matrix_coordinates(
        df_tracking_passes, frame_col=unique_frame_col, player_col=tracking_player_col,
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
    import streamlit as st
    if exclude_passer:
        passers_to_exclude = df_passes[event_player_col].values  # F
    else:
        passers_to_exclude = None

    # 5. Simulate passes to get expected completion
    simulation_result = simulate_passes_chunked(
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

        # Chunk size
        chunk_size=chunk_size,
        log=False,
    )
    if use_poss:
        xc = simulation_result.poss_cum_att[:, 0, -1]  # F x PHI x T ---> F
    else:
        xc = simulation_result.prob_cum_att[:, 0, -1]  # F x PHI x T ---> F

    # outcomes = df_passes[outcome_col].values
    # brier = sklearn.metrics.brier_score_loss(outcomes, xc)
    # logloss = sklearn.metrics.log_loss(outcomes, xc, labels=[0, 1])
    # average_completion = np.mean(outcomes)
    # brier_baseline = np.mean((outcomes - average_completion)**2)
    # brier_skill_score = 1 - brier / brier_baseline

    idx = df_passes[event_frame_col].map(frame_to_idx)

    return xc, idx, simulation_result


def _dist_to_opp_goal(x_norm, y_norm):
    MAX_GOAL_POST_RADIUS = 0.06
    SEMI_GOAL_WIDTH_INNER_EDGE = 7.32 / 2
    SEMI_GOAL_WIDTH_CENTER = SEMI_GOAL_WIDTH_INNER_EDGE + MAX_GOAL_POST_RADIUS
    def _distance(x, y, x_target, y_target):
        return np.sqrt((x - x_target) ** 2 + (y - y_target) ** 2)
    x_goal = 52.5
    y_goal = np.clip(y_norm, -SEMI_GOAL_WIDTH_CENTER, SEMI_GOAL_WIDTH_CENTER)
    return _distance(x_norm, y_norm, x_goal, y_goal)


def _opening_angle_to_goal(x, y):
    MAX_GOAL_POST_RADIUS = 0.06
    SEMI_GOAL_WIDTH_INNER_EDGE = 7.32 / 2
    SEMI_GOAL_WIDTH_CENTER = SEMI_GOAL_WIDTH_INNER_EDGE + MAX_GOAL_POST_RADIUS

    def angle_between(u, v):
        divisor = np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0)
        i_div_0 = divisor == 0
        divisor[i_div_0] = np.inf  # Avoid division by zero by setting divisor to inf
        dot_product = np.sum(u * v, axis=0)
        cosTh1 = dot_product / divisor
        angle = np.arccos(cosTh1)
        return angle

    x_goal = 52.5
    return np.abs(angle_between(np.array([x_goal - x, SEMI_GOAL_WIDTH_CENTER - y]), np.array([x_goal - x, -SEMI_GOAL_WIDTH_CENTER - y])))


def _get_danger(dist_to_goal, opening_angle):
    coefficients = [-0.14447723, 0.40579492]
    intercept = -0.52156283
    logit = intercept + coefficients[0] * dist_to_goal + coefficients[1] * opening_angle
    prob_true = 1 / (1 + np.exp(-logit))
    return prob_true


def get_dangerous_accessible_space(
    # Data
    df_tracking, tracking_frame_col="frame_id", tracking_player_col="player_id", tracking_team_col="team_id",
    ball_tracking_player_id="ball", tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy",
    attacking_direction_col="attacking_direction",

    # Options
    apply_pitch_mask_to_raw_result=False,

    # Parameters
    n_angles=_DEFAULT_N_ANGLES_FOR_DAS,
    phi_offset=_DEFAULT_PHI_OFFSET,
    n_v0=_DEFAULT_N_V0_FOR_DAS,
    v0_min=_DEFAULT_V0_MIN_FOR_DAS,
    v0_max=_DEFAULT_V0_MAX_FOR_DAS,
):
    PLAYER_POS, BALL_POS, players, player_teams, controlling_teams, frame_to_idx = get_matrix_coordinates(
        df_tracking, frame_col=tracking_frame_col, player_col=tracking_player_col,
        ball_player_id=ball_tracking_player_id, team_col=tracking_team_col, x_col=tracking_x_col, y_col=tracking_y_col,
        vx_col=tracking_vx_col, vy_col=tracking_vy_col,
    )
    F = PLAYER_POS.shape[0]

    phi_grid = np.tile(np.linspace(phi_offset, 2*np.pi+phi_offset, n_angles, endpoint=False), (F, 1))  # F x PHI
    v0_grid = np.tile(np.linspace(v0_min, v0_max, n_v0), (F, 1))  # F x V0

    simulation_result = simulate_passes_chunked(
        PLAYER_POS, BALL_POS, phi_grid, v0_grid, controlling_teams, player_teams, players, passers_to_exclude=None,
    )
    if apply_pitch_mask_to_raw_result:
        simulation_result = mask_out_of_play(simulation_result)

    # Add danger to simulation result
    if attacking_direction_col is not None:
        fr2playingdirection = df_tracking[[tracking_frame_col, attacking_direction_col]].set_index(tracking_frame_col).to_dict()[attacking_direction_col]
        ATTACKING_DIRECTION = np.array([fr2playingdirection[frame] for frame in frame_to_idx])  # F
    else:
        ATTACKING_DIRECTION = np.ones(F)
    X = simulation_result.x_grid
    Y = simulation_result.y_grid
    X_NORM = X * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]
    Y_NORM = Y * ATTACKING_DIRECTION[:, np.newaxis, np.newaxis]
    DIST_TO_GOAL = _dist_to_opp_goal(X_NORM, Y_NORM)
    OPENING_ANGLE = _opening_angle_to_goal(X_NORM, Y_NORM)
    DANGER = _get_danger(DIST_TO_GOAL, OPENING_ANGLE)
    simulation_result = simulation_result._replace(danger=DANGER)

    # Get AS and DAS
    accessible_space = aggregate_surface_area(simulation_result)  # F
    das = aggregate_surface_area(simulation_result, add_danger=True)  # F
    fr2AS = pd.Series(accessible_space, index=df_tracking[tracking_frame_col].unique())
    fr2DAS = pd.Series(das, index=df_tracking[tracking_frame_col].unique())
    as_series = df_tracking[tracking_frame_col].map(fr2AS)
    das_series = df_tracking[tracking_frame_col].map(fr2DAS)

    idx = df_tracking[tracking_frame_col].map(frame_to_idx)

    return as_series, das_series, idx, simulation_result


def infer_playing_direction(
    df_tracking, team_col="team_id", period_col="period_id", possession_team_col="ball_possession", x_col="x",
):
    """ Automatically infer playing direction based on the mean x position of each teams in each period. """
    playing_direction = {}
    for period_id, df_tracking_period in df_tracking.groupby(period_col):
        x_mean = df_tracking_period.groupby(team_col)[x_col].mean()
        smaller_x_team = x_mean.idxmin()
        greater_x_team = x_mean.idxmax()
        playing_direction[period_id] = {smaller_x_team: 1, greater_x_team: -1}

    new_attacking_direction = pd.Series(index=df_tracking.index, dtype=np.float64)

    for period_id in playing_direction:
        i_period = df_tracking[period_col] == period_id
        for team_id, direction in playing_direction[period_id].items():
            i_period_team_possession = i_period & (df_tracking[possession_team_col] == team_id)
            new_attacking_direction.loc[i_period_team_possession] = direction

    return new_attacking_direction


def _adjust_saturation(color, saturation):
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, l, saturation)


def plot_expected_completion_surface(
    das_simulation_result, frame_index, plot_type_off="poss", plot_type_def=None, color_off="blue", color_def="red",
    plot_gridpoints=True
):
    x_grid = das_simulation_result.x_grid[frame_index, :, :]
    y_grid = das_simulation_result.y_grid[frame_index, :, :]

    x = np.ravel(x_grid)  # F*PHI*T
    y = np.ravel(y_grid)  # F*PHI*T

    for offdef, plot_type, color in [("off", plot_type_off, color_off), ("def", plot_type_def, color_def)]:
        if plot_type is None:
            continue
        if offdef == "off":
            if plot_type == "poss":
                p = das_simulation_result.poss_density_att[frame_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_att[frame_index, :, :]
            else:
                raise ValueError(f"Unknown plot type: {plot_type}. Must be 'poss' or 'prob'.")
        else:
            if plot_type == "poss":
                p = das_simulation_result.poss_density_def[frame_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_def[frame_index, :, :]
            else:
                raise ValueError(f"Unknown plot type: {plot_type}. Must be 'poss' or 'prob'.")

        z = np.ravel(p)  # F*PHI*T

        areas = 10
        absolute_scale = False
        if absolute_scale:
            levels = np.linspace(start=0, stop=1.1, num=areas + 1, endpoint=True)
        else:
            levels = np.linspace(start=0, stop=np.max(z)+0.00001, num=areas + 1, endpoint=True)
        saturations = [x / (areas) for x in range(areas)]
        base_color = matplotlib.colors.to_rgb(color)

        colors = [_adjust_saturation(base_color, s) for s in saturations]

        # Create a triangulation
        triang = matplotlib.tri.Triangulation(x, y)
        cp = plt.tricontourf(x, y, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale
        plt.tricontourf(triang, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale

    if plot_gridpoints:
        plt.plot(x, y, 'ko', ms=0.5)

    return plt.gcf()
