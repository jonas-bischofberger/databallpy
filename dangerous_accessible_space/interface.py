import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np

import assets.test_data
import streamlit as st

import dangerous_accessible_space.core


def get_matrix_coordinates(
    df_tracking,
    frame_col="frame_id",
    player_col="player_id",
    ball_player_id="ball",
    team_col="team_id",
    x_col="x",
    y_col="y",
    vx_col="vx",
    vy_col="vy"
):
    """
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
    >>> PLAYER_POS.shape, BALL_POS.shape, player_list.shape, team_indices.shape
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

    player_list = df_players.columns.get_level_values(1).unique()  # P
    player2team = df_tracking.loc[i_player, [player_col, team_col]].drop_duplicates().set_index(player_col)[team_col]
    team_indices = player2team.loc[player_list].values

    df_ball = df_tracking.loc[~i_player].set_index(frame_col)[[x_col, y_col, vx_col, vy_col]]
    BALL_POS = df_ball.values  # F x C

    return PLAYER_POS, BALL_POS, player_list, team_indices


def _get_unused_column_name(df, prefix):
    i = 1
    new_column_name = prefix
    while new_column_name in df.columns:
        new_column_name = f"{prefix}_{i}"
        i += 1
    return new_column_name


def get_pass_velocity(df_passes, df_tracking_ball, event_frame_col="frame_id", tracking_frame_col="frame_id", n_frames_after_pass_for_v0=5, fallback_v0=10, vx_col="vx", vy_col="vy", v_col=None):
    """
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
    if v_col is not None:
        df_tracking_ball_v0[ball_velocity_col] = df_tracking_ball_v0[v_col]
    else:
        df_tracking_ball_v0[ball_velocity_col] = np.sqrt(df_tracking_ball_v0[vx_col] ** 2 + df_tracking_ball_v0[vy_col] ** 2)

    dfg_v0 = df_tracking_ball_v0.groupby(pass_nr_col)[ball_velocity_col].mean()

    v0 = df_passes[pass_nr_col].map(dfg_v0)
    v0 = v0.fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames
    return v0


if __name__ == '__main__':
    # get_matrix_coordinates(assets.test_data.df_tracking)
    assets.test_data.df_passes["v0"] = get_pass_velocity(assets.test_data.df_passes, assets.test_data.df_tracking[assets.test_data.df_tracking["player_id"] == "ball"])
    st.write("assets.test_data.df_passes")
    st.write(assets.test_data.df_passes)


def get_expected_pass_completion(
    df_passes,
    df_tracking,
):
    ### 1. Prepare data
    # 1.1 Extract player positions
    # x_cols = [col for col in df_tracking_passes.columns if col.endswith("_x_norm") and not (col.startswith("start") or col.startswith("end"))]
    # x_cols_players = [col for col in x_cols if "ball" not in col]
    #
    # coordinates = ["_x_norm", "_y_norm", "_vx_norm", "_vy_norm"]
    #
    # df_coords = df_tracking_passes[[f"{x_col.replace('_x_norm', coord)}" for x_col in x_cols_players for coord in coordinates]]
    #
    # st.write("df_coords")
    # st.write(df_coords)
    #
    # F = df_coords.shape[0]  # number of frames
    # C = len(coordinates)
    # P = df_coords.shape[1] // len(coordinates)
    # PLAYER_POS = df_coords.values.reshape(F, P, C)#.transpose(1, 0, 2)  # F x P x C
    # st.write("PLAYER_POS", PLAYER_POS.shape)
    # st.write(PLAYER_POS[0, :, :])
    PLAYER_POS, BALL_POS, player_list, team_list = get_matrix_coordinates(df_tracking_passes)

    # 1.2 Extract ball position
    # BALL_POS = df_tracking_passes[[f"ball{coord}" for coord in coordinates]].values  # F x C
    # st.write("BALL_POS", BALL_POS.shape)
    # st.write(BALL_POS)

    # 1.3 Extract v0 as mean ball_velocity of the first N frames after the pass
    df_tracking_passes = _add_pass_velocity(df_tracking_passes, df_tracking_and_event)

    # df_tracking_passes["pass_nr"] = df_tracking_passes.index
    # index = [[idx + i for i in range(n_frames_after_pass_for_v0)] for idx in df_tracking_passes.index]
    # index = [item for sublist in index for item in sublist]
    # df_tracking_v0 = df_tracking_and_event.loc[index]
    # df_tracking_v0["related_pass_id"] = df_tracking_v0["pass_id"].ffill()
    # dfg_v0 = df_tracking_v0.groupby("related_pass_id")["ball_velocity"].mean()
    # df_tracking_passes["v0"] = df_tracking_passes["pass_id"].map(dfg_v0)
    # df_tracking_passes["v0"] = df_tracking_passes["v0"].fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames

    v0_grid = df_tracking_passes["v0"].values.repeat(30).reshape(-1, 30)  # F x V0

    # 1.4 Extract starting angle (phi)
    df_tracking_passes["phi"] = np.arctan2(df_tracking_passes["end_y_norm"] - df_tracking_passes["start_y_norm"], df_tracking_passes["end_x_norm"] - df_tracking_passes["start_x_norm"])

    st.write("df_tracking_passes", df_tracking_passes.shape)
    st.write(df_tracking_passes)
    st.write("v0_grid", v0_grid.shape)
    st.write(v0_grid)

    phi_grid = df_tracking_passes["phi"].values[:, np.newaxis]  # F x PHI

    # 1.5 Extract player team info

    passer_team = df_tracking_passes["team"].values  # F
    st.write("passer_team", passer_team.shape)
    st.write(passer_team)

    ball_possession = df_tracking_passes["ball_possession"].values  # F
    st.write("ball_possession", ball_possession.shape)
    st.write(ball_possession)

    i_not_same = passer_team != ball_possession

    if i_not_same.any():
        st.warning(f"Passer team and tracking ball possession team are not the same for {i_not_same.sum()} passes. Prefer event info.")

    # player_list = [col.split("_x_norm")[0] for col in df_tracking_passes.columns if col.endswith("_x_norm") and not (col.startswith("start") or col.startswith("end") or "ball" in col)]  # P
    # st.write("player_list", len(player_list))
    # st.write(player_list)

    # team_list = np.array(["home" if "home" in player else "away" for player in player_list])  # P

    player_list = np.array(player_list)  # P
    team_list = np.array(team_list)  # P

    simulation_result = simulate_passes(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list)

    xc = simulation_result.p_cum_att[:, 0, -1]  # F x PHI x T ---> F
    df_tracking_passes["xC"] = xc

    st.write("df_tracking_passes")
    st.write(df_tracking_passes)

    brier = np.mean(df_tracking_passes["outcome"] - df_tracking_passes["xC"])**2
    logloss = sklearn.metrics.log_loss(df_tracking_passes["outcome"], df_tracking_passes["xC"])
    average_completion_rate = np.mean(df_tracking_passes["outcome"])
    st.write("average_completion_rate")
    st.write(average_completion_rate)

    brier_baseline = np.mean((df_tracking_passes["outcome"] - average_completion_rate)**2)

    st.write("brier", brier)
    st.write("logloss", logloss)
    st.write("brier_baseline", brier_baseline)

    xc = df_tracking_passes["xC"]

    df_tracking_passes["xC_string"] = xc.apply(lambda x: f"xC={x:.1%}")

    st.write("xc", xc)

    for pass_nr, (pass_index, p4ss) in enumerate(df_tracking_passes.iterrows()):
        if pass_nr < 4:
            continue

        fig, ax = databallpy.visualize.plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
        fig, ax = databallpy.visualize.plot_tracking_data(
            match,
            pass_index,
            fig=fig,
            ax=ax,
            # events=["pass"],
            title="First pass after the kick-off",
            add_velocities=True,
            variable_of_interest=df_tracking_passes.loc[pass_index, "xC_string"],
        )
        # st.write("p4ss")
        # st.write(p4ss)
        plt.arrow(
            p4ss["start_x_norm"], p4ss["start_y_norm"], p4ss["end_x_norm"] - p4ss["start_x_norm"],
            p4ss["end_y_norm"] - p4ss["start_y_norm"], head_width=1, head_length=1, fc='blue', ec='blue'
        )
        st.write(fig)
        plt.close(fig)

        break

    return df_tracking_passes
