import streamlit_profiler
profiler = streamlit_profiler.Profiler()
profiler.start()


import databallpy.features
import databallpy.visualize
import databallpy
import streamlit as st
import collections
import math
import numpy as np
import scipy
import pandas as pd
import time
from typing import Literal
import sklearn.metrics

import matplotlib.pyplot as plt

import dangerous_accessible_space


@st.cache_resource
def get_data():
    match = databallpy.get_open_match()
    return match


@st.cache_resource
def get_preprocessed_data():
    match = get_data()
    match.synchronise_tracking_and_event_data()
    databallpy.features.add_team_possession(match.tracking_data, match.event_data, match.home_team_id, inplace=True)

    databallpy.features.add_velocity(match.tracking_data, inplace=True,
                                     column_ids=match.home_players_column_ids() + match.away_players_column_ids(),
                                     frame_rate=match.frame_rate)

    return match


def _add_attacking_direction_normalized_coordinates(df_tracking, match, col_suffix="_norm"):
    tracking_player_ids = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
    position_cols = [f"{tracking_player_id}_{coord}" for tracking_player_id in tracking_player_ids for coord in ["x", "y", "vx", "vy"]] + ["start_x", "start_y", "end_x", "end_y"]
    position_cols_norm = [f"{col}{col_suffix}" for col in position_cols]
    i_away_possession = df_tracking["ball_possession"] == "away"
    df_tracking[position_cols_norm] = df_tracking[position_cols]
    df_tracking.loc[i_away_possession, position_cols_norm] = -df_tracking.loc[i_away_possession, position_cols_norm]
    return df_tracking


@st.cache_resource
def _get_preprocessed_tracking_and_event_data():
    match = get_preprocessed_data()

    df_tracking = match.tracking_data
    players = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
    dft = df_tracking[["frame"] + [f"{player}_{coord}" for player in players for coord in ["x", "y", "vx", "vy"]]]

    player2team = {}
    for player in players:
        if player in match.home_players_column_ids():
            player2team[player] = match.home_team_id
        elif player in match.away_players_column_ids():
            player2team[player] = match.away_team_id
        else:
            player2team[player] = None

    dfs_player = []
    for player in players:
        df_player = dft[["frame"] + [f"{player}_{coord}" for coord in ["x", "y", "vx", "vy"]]].rename(columns={f"{player}_{coord}": coord for coord in ["x", "y", "vx", "vy"]})
        df_player["player_id"] = player
        df_player["team_id"] = player2team[player]
        dfs_player.append(df_player)
        dft = dft.drop(columns=[f"{player}_{coord}" for coord in ["x", "y", "vx", "vy"]], axis=1)
    df_player = pd.concat(dfs_player, axis=0)

    dft = dft.merge(df_player, on="frame", how="left")

    df_event = match.event_data.rename(columns={"td_frame": "frame"})

    return dft, df_event


def get_expected_pass_completion(match):
    df_tracking, df_event = _get_preprocessed_tracking_and_event_data()

    df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()
    df_passes = df_passes.rename(columns={"tracking_frame": "frame"})

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
    i_pass_in_tracking = df_tracking["frame"].isin(df_passes["frame"])

    PLAYER_POS, BALL_POS, player_list, team_list = dangerous_accessible_space.get_matrix_coordinates(df_tracking.loc[i_pass_in_tracking], frame_col="frame", player_col="player_id")

    # 1.2 Extract ball position
    # BALL_POS = df_tracking_passes[[f"ball{coord}" for coord in coordinates]].values  # F x C
    # st.write("BALL_POS", BALL_POS.shape)
    # st.write(BALL_POS)

    # 1.3 Extract v0 as mean ball_velocity of the first N frames after the pass
    st.write("df_passes A")
    st.write(df_passes)
    df_passes["v0"] = dangerous_accessible_space.get_pass_velocity(df_passes, df_tracking[df_tracking["player_id"] == "ball"], frame_col="frame")

    st.write("df_passes B")
    st.write(df_passes)
    st.write("df_tracking")
    st.write(df_tracking.head(5000))

    # df_tracking_passes["pass_nr"] = df_tracking_passes.index
    # index = [[idx + i for i in range(n_frames_after_pass_for_v0)] for idx in df_tracking_passes.index]
    # index = [item for sublist in index for item in sublist]
    # df_tracking_v0 = df_tracking_and_event.loc[index]
    # df_tracking_v0["related_pass_id"] = df_tracking_v0["pass_id"].ffill()
    # dfg_v0 = df_tracking_v0.groupby("related_pass_id")["ball_velocity"].mean()
    # df_tracking_passes["v0"] = df_tracking_passes["pass_id"].map(dfg_v0)
    # df_tracking_passes["v0"] = df_tracking_passes["v0"].fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames

    v0_grid = df_passes["v0"].values.repeat(30).reshape(-1, 30)  # F x V0

    # 1.4 Extract starting angle (phi)
    df_passes["phi"] = np.arctan2(df_passes["end_y"] - df_passes["start_y"], df_passes["end_x"] - df_passes["start_x"])

    st.write("df_passes", df_passes.shape)
    st.write(df_passes)
    st.write("v0_grid", v0_grid.shape)
    st.write(v0_grid)

    phi_grid = df_passes["phi"].values[:, np.newaxis]  # F x PHI

    st.write("df_passes")
    st.write(df_passes)

    # 1.5 Extract player team info
    passer_team = df_passes["team_id"].values  # F
    st.write("passer_team", passer_team.shape)
    st.write(passer_team)

    # ball_possession = df_tracking_passes["ball_possession"].values  # F
    # st.write("ball_possession", ball_possession.shape)
    # st.write(ball_possession)

    # player_list = [col.split("_x_norm")[0] for col in df_tracking_passes.columns if col.endswith("_x_norm") and not (col.startswith("start") or col.startswith("end") or "ball" in col)]  # P
    # st.write("player_list", len(player_list))
    # st.write(player_list)

    # team_list = np.array(["home" if "home" in player else "away" for player in player_list])  # P

    player_list = np.array(player_list)  # P
    team_list = np.array(team_list)  # P

    simulation_result = dangerous_accessible_space.simulate_passes(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list)

    xc = simulation_result.p_cum_att[:, 0, -1]  # F x PHI x T ---> F
    df_passes["xC"] = xc

    st.write("df_passes")
    st.write(df_passes)

    brier = np.mean(df_passes["outcome"] - df_passes["xC"])**2
    logloss = sklearn.metrics.log_loss(df_passes["outcome"], df_passes["xC"])
    average_completion_rate = np.mean(df_passes["outcome"])
    st.write("average_completion_rate")
    st.write(average_completion_rate)

    brier_baseline = np.mean((df_passes["outcome"] - average_completion_rate)**2)

    st.write("brier", brier)
    st.write("logloss", logloss)
    st.write("brier_baseline", brier_baseline)

    xc = df_passes["xC"]

    df_passes["xC_string"] = xc.apply(lambda x: f"xC={x:.1%}")

    st.write("xc", xc)

    for pass_nr, (pass_index, p4ss) in enumerate(df_passes.iterrows()):
        fig, ax = databallpy.visualize.plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
        fig, ax = databallpy.visualize.plot_tracking_data(
            match,
            p4ss["frame"],
            fig=fig,
            ax=ax,
            team_colors=["blue", "red"],
            # events=["pass"],
            title="First pass after the kick-off",
            add_velocities=True,
            variable_of_interest=df_passes.loc[pass_index, "xC_string"],
        )
        st.write("p4ss")
        st.write(df_passes)
        st.write("match.passes_df.loc[pass_index]")
        st.write(match.passes_df)
        plt.arrow(
            p4ss["start_x"], p4ss["start_y"], p4ss["end_x"] - p4ss["start_x"],
            p4ss["end_y"] - p4ss["start_y"], head_width=1, head_length=1, fc='blue', ec='blue'
        )
        st.write(fig)
        plt.close(fig)

        if pass_nr > 5:
            break

    return df_passes


#     PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
#     BALL_POS,  # F x 2[x, y], ball positions
#     phi_grid,  # F x PHI, pass angles
#     v0_grid,  # F x V0, pass speeds
#     passer_team,  # F, team of passers
#     team_list,  # P, player teams


def plot_expected_completion_surface(simulation_result, fr, plot_gridpoints=True):
    # simulation_result = mask_out_of_play(simulation_result)

    p = simulation_result.p_density_att[fr, :, :]  # F x PHI x T
    # simulation_result.on_pitch_mask # phi_grid, on_pitch_mask

    # st.write(simulation_result.x0_grid.shape)  # F
    # st.write(simulation_result.r_grid.shape)  # T
    # st.write(simulation_result.phi_grid.shape)  # F x PHI

    # x_grid = simulation_result.x0_grid[:, np.newaxis, np.newaxis] + simulation_result.r_grid[np.newaxis, np.newaxis, :] * np.cos(simulation_result.phi_grid[:, :, np.newaxis])  # F x PHI x T
    # y_grid = simulation_result.y0_grid[:, np.newaxis, np.newaxis] + simulation_result.r_grid[np.newaxis, np.newaxis, :] * np.sin(simulation_result.phi_grid[:, :, np.newaxis])  # F x PHI x T
    #
    # x_grid = x_grid[fr, :, :]
    # y_grid = y_grid[fr, :, :]

    x_grid = simulation_result.x_grid[fr, :, :]
    y_grid = simulation_result.y_grid[fr, :, :]

    st.write(f"x_grid[fr={fr}, phi=:, T=:]", x_grid.shape)
    st.write(x_grid)
    st.write(f"y_grid[fr={fr}, phi=:, T=:]", y_grid.shape)
    st.write(y_grid)
    st.write(f"p[fr={fr}, phi=:, T=:]", p.shape)
    st.write(p)

    x = np.ravel(x_grid)  # F*PHI*T
    y = np.ravel(y_grid)  # F*PHI*T
    z = p
    z = np.ravel(z)  # F*PHI*T

    z = np.minimum(z, 0.9)

    areas = 10
    absolute_scale = False
    if absolute_scale:
        levels = np.linspace(start=0, stop=1.1, num=areas + 1, endpoint=True)
    else:
        levels = np.linspace(start=0, stop=np.max(z)+0.00001, num=areas + 1, endpoint=True)
    saturations = [x / (areas) for x in range(areas)]
    import matplotlib.colors
    base_color = matplotlib.colors.to_rgb("blue")
    def adjust_saturation(color, saturation):
        import colorsys
        h, l, s = colorsys.rgb_to_hls(*color)
        return colorsys.hls_to_rgb(h, l, saturation)

    colors = [adjust_saturation(base_color, s) for s in saturations]

    # Create a triangulation
    import matplotlib.tri
    triang = matplotlib.tri.Triangulation(x, y)
    cp = plt.tricontourf(x, y, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale
    plt.tricontourf(triang, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale

    if plot_gridpoints:
        plt.plot(x, y, 'ko', ms=0.5)


def aggregate_surface_area(result):
    st.write("result")
    st.write(result._fields)

    result = dangerous_accessible_space.mask_out_of_play(result)

    # Get r-part of area elements
    r_grid = result.r_grid  # T

    r_lower_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_lower_bounds[1:] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and previous element
    r_lower_bounds[0] = r_grid[0]  # Set lower bound for the first element

    r_upper_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_upper_bounds[:-1] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and next element
    r_upper_bounds[-1] = r_grid[-1]  # Arbitrarily high upper bound for the last element

    dr = r_upper_bounds - r_lower_bounds  # T
    st.write("dr", dr.shape)
    st.write(dr)

    # Get phi-part of area elements
    phi_grid = result.phi_grid  # F x PHI
    st.write("phi_grid", phi_grid.shape)
    st.write(phi_grid)

    # dA = np.diff(phi_grid, axis=1) * r_grid[:, np.newaxis]  # F x PHI-1 x T

    phi_lower_bounds = np.zeros_like(phi_grid)  # F x PHI
    phi_lower_bounds[:, 1:] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and previous element
    phi_lower_bounds[:, 0] = phi_grid[:, 0]

    phi_upper_bounds = np.zeros_like(phi_grid)  # Initialize with zeros
    phi_upper_bounds[:, :-1] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and next element
    phi_upper_bounds[:, -1] = phi_grid[:, -1]  # Arbitrarily high upper bound for the last element

    st.write("phi_lower_bounds", phi_lower_bounds.shape)
    st.write(phi_lower_bounds)

    st.write("phi_upper_bounds", phi_upper_bounds.shape)
    st.write(phi_upper_bounds)

    dphi = phi_upper_bounds - phi_lower_bounds  # F x PHI

    outer_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_upper_bounds[np.newaxis, np.newaxis, :]**2)  # T
    inner_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_lower_bounds[np.newaxis, np.newaxis, :]**2)  # T

    dA = outer_bound_circle_slice_area - inner_bound_circle_slice_area  # F x PHI x T

    p = result.p_density_att * dr
    st.write("result.p_density_att", result.p_density_att.shape)
    st.write(result.p_density_att[0, :, :])
    st.write("p", p.shape)
    st.write(p[0, :, :])
    AS = np.sum(p * dA, axis=(1, 2))  # F

    return AS

    # xt = get_xT_prediction(x, y, THROW_IN_XT)

    st.stop()


def get_dangerous_accessible_space(match):
    fig, ax = databallpy.visualize.plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")

    df_tracking, df_events = _get_preprocessed_tracking_and_event_data()

    df_passes = df_events[df_events["databallpy_event"] == "pass"].reset_index().iloc[:10]
    df_tracking = df_tracking[df_tracking["frame"].isin(df_passes["frame"])]
    valid_frames = df_tracking[(df_tracking["player_id"] == "ball") & (df_tracking["x"].notna())]["frame"]
    df_passes = df_passes[df_passes["frame"].isin(valid_frames)]

    st.write("df_tracking")
    st.write(df_tracking)
    st.write("df_passes")
    st.write(df_passes)

    import dangerous_accessible_space
    PLAYER_POS, BALL_POS, player_list, team_list = dangerous_accessible_space.get_matrix_coordinates(df_tracking, frame_col="frame", player_col="player_id")

    F = PLAYER_POS.shape[0]

    n_angles = 3#50
    phi_offset = 0#-math.pi/16
    n_v0 = 20

    phi_grid = np.tile(np.linspace(phi_offset, 2*np.pi+phi_offset, n_angles, endpoint=False), (F, 1))  # F x PHI

    # st.write("phi_grid", phi_grid.shape)
    # st.write(phi_grid)

    v0_grid = np.tile(np.linspace(3, 40, n_v0), (F, 1))  # F x V0
    # passer_team = df_tracking_and_event["ball_possession"].values  # F
    passer_team = team_list

    # this should be returned by _get_matrix_coordinates
    # team_list = np.array(["home" if "home" in player else "away" for player in match.home_players_column_ids() + match.away_players_column_ids()])  # P

    simulation_result = dangerous_accessible_space.simulate_passes_chunked(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list)
    # simulation_result = mask_out_of_play(simulation_result)
    st.write("B")

    # simulation_result_das = dangerous_accessible_space.add_xT_to_result(simulation_result)

    fig, ax = databallpy.visualize.plot_tracking_data(
        match,
        df_tracking.iloc[0]["frame"],
        fig=fig,
        ax=ax,
        events=["pass"],
        add_velocities=True,
    )
    # plot_expected_completion_surface(simulation_result_das, 0)
    plot_expected_completion_surface(simulation_result, 0)
    st.write(fig)

    accessible_space = aggregate_surface_area(simulation_result)
    st.write("AS", accessible_space)
    df_passes["AS"] = accessible_space
    # dangerous_accessible_space = aggregate_surface_area(simulation_result_das)
    # st.write("DAS", dangerous_accessible_space)
    # df_passes["DAS"] = dangerous_accessible_space

    return df_passes

    # return simulation_result


#     PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
#     BALL_POS,  # F x 2[x, y], ball positions
#     phi_grid,  # F x PHI, pass angles
#     v0_grid,  # F x V0, pass speeds
#     passer_team,  # F, team of passers
#     team_list,  # P, player teams

def main():
    match = get_preprocessed_data()

    st.write("match.event_data")
    st.write(match.event_data)
    st.write("match.tracking_data")
    st.write(match.tracking_data.head(500))

    # fig, ax = databallpy.visualize.plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
    # fig, ax = databallpy.visualize.plot_tracking_data(
    #     match,
    #     10000,
    #     fig=fig,
    #     ax=ax,
    #     events=["pass"],
    #     title="First pass after the kick-off",
    #     add_velocities=True,
    #     variable_of_interest="test",
    # )
    # st.write(fig)

    df_tracking, df_event = _get_preprocessed_tracking_and_event_data()

    # df_tracking_passes = get_expected_pass_completion(match)
    # st.write("result df_tracking_passes")
    # st.write(df_tracking_passes)

    df_das = get_dangerous_accessible_space(match)
    st.write("result df_das")
    st.write(df_das)

    profiler.stop()


if __name__ == '__main__':
    main()
