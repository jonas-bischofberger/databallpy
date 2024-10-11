import streamlit_profiler
import tqdm

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
import colorsys

import matplotlib.pyplot as plt
import sklearn.model_selection

import dangerous_accessible_space

import importlib
importlib.reload(dangerous_accessible_space)


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


def _adjust_saturation(color, saturation):
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, l, saturation)


def plot_expected_completion_surface(
    das_simulation_result, F_index, plot_type_off="poss", plot_type_def=None, color_off="blue", color_def="red",
    plot_gridpoints=True
):
    x_grid = das_simulation_result.x_grid[F_index, :, :]
    y_grid = das_simulation_result.y_grid[F_index, :, :]

    x = np.ravel(x_grid)  # F*PHI*T
    y = np.ravel(y_grid)  # F*PHI*T

    for offdef, plot_type, color in [("off", plot_type_off, color_off), ("def", plot_type_def, color_def)]:
        if plot_type is None:
            continue
        if offdef == "off":
            if plot_type == "poss":
                p = das_simulation_result.poss_density_att[F_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_att[F_index, :, :]
            else:
                raise ValueError(f"Unknown plot type: {plot_type}. Must be 'poss' or 'prob'.")
        else:
            if plot_type == "poss":
                p = das_simulation_result.poss_density_def[F_index, :, :]
            elif plot_type == "prob":
                p = das_simulation_result.prob_density_def[F_index, :, :]
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
        import matplotlib.colors
        base_color = matplotlib.colors.to_rgb(color)

        colors = [_adjust_saturation(base_color, s) for s in saturations]

        # Create a triangulation
        import matplotlib.tri
        triang = matplotlib.tri.Triangulation(x, y)
        cp = plt.tricontourf(x, y, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale
        plt.tricontourf(triang, z.T, colors=colors, alpha=0.1, cmap=None, levels=levels)  # Comment in to use [0, 1] scale

    if plot_gridpoints:
        plt.plot(x, y, 'ko', ms=0.5)

    return plt.gcf()


@st.cache_resource
def _get_preprocessed_data():
    match = get_preprocessed_data()

    players = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
    x_cols = [f"{player}_x" for player in players]
    y_cols = [f"{player}_y" for player in players]
    vx_cols = [f"{player}_vx" for player in players]
    vy_cols = [f"{player}_vy" for player in players]
    v_cols = [f"{player}_velocity" for player in players]
    frame_col = "frame"

    player_to_team = {}
    for player in players:
        if player in match.home_players_column_ids():
            player_to_team[player] = match.home_team_id
        elif player in match.away_players_column_ids():
            player_to_team[player] = match.away_team_id
        else:
            player_to_team[player] = None

    df_tracking = dangerous_accessible_space.per_object_frameify_tracking_data(match.tracking_data, frame_col, x_cols, y_cols, vx_cols, vy_cols, players, player_to_team, v_cols=v_cols)
    df_tracking["ball_possession"] = df_tracking["ball_possession"].map({"home": match.home_team_id, "away": match.away_team_id})

    df_events = match.event_data
    df_events["tracking_player_id"] = df_events["player_id"].map(match.player_id_to_column_id)

    return match, df_tracking, df_events


def validate(n_steps=100, training_size=0.99):
    match, df_tracking, df_event = _get_preprocessed_data()
    df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()

    random_state = 1893
    df_training, df_test = sklearn.model_selection.train_test_split(df_passes, train_size=training_size, random_state=random_state)

    df_training = df_passes
    df_test = df_passes

    def _choose_random_parameters(parameter_to_bounds):
        random_parameters = {}
        for param, bounds in parameter_to_bounds.items():
            # st.write("B", param, bounds, str(type(bounds[0])), str(type(bounds[-1])), "bool", isinstance(bounds[0], bool), isinstance(bounds[0], int), isinstance(bounds[0], float))
            if isinstance(bounds[0], bool):  # order matters, bc bool is also int
                random_parameters[param] = np.random.choice([bounds[0], bounds[-1]])
            elif isinstance(bounds[0], int) or isinstance(bounds[0], float):
                random_parameters[param] = np.random.uniform(bounds[0], bounds[-1])
            else:
                raise NotImplementedError(f"Unknown type: {type(bounds[0])}")
        return random_parameters

    data = {
        "brier": [],
        "logloss": [],
        "auc": [],
        "parameters": [],
    }
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    display_df = st.empty()
    for i in tqdm.tqdm(range(n_steps), desc="Simulation", total=n_steps):
        progress_bar_text.text(f"Simulation {i+1}/{n_steps}")
        progress_bar.progress((i+1) / n_steps)
        random_paramter_assignment = _choose_random_parameters(dangerous_accessible_space.PARAMETER_BOUNDS)
        xc, _, _ = dangerous_accessible_space.get_expected_pass_completion(
            df_training, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
            event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
            event_player_col="tracking_player_id",
            **random_paramter_assignment,
        )
        brier = sklearn.metrics.brier_score_loss(df_training["outcome"], xc)
        logloss = sklearn.metrics.log_loss(df_training["outcome"], xc)
        auc = sklearn.metrics.roc_auc_score(df_training["outcome"], xc)
        data["brier"].append(brier)
        data["logloss"].append(logloss)
        data["auc"].append(auc)
        data["parameters"].append(random_paramter_assignment)
        display_df.write(pd.DataFrame(data).sort_values("logloss"), ascending=True)

    df_training_results = pd.DataFrame(data)
    st.write("df_training_results")
    st.write(df_training_results)

    best_index = df_training_results["logloss"].idxmin()
    best_parameters = df_training_results["parameters"][best_index]
    st.write("Best parameters")
    st.write(best_parameters)

    xc, _, _ = dangerous_accessible_space.get_expected_pass_completion(
        df_test, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
        event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
        event_player_col="tracking_player_id",
        **best_parameters,
    )
    brier = sklearn.metrics.brier_score_loss(df_test["outcome"], xc)
    logloss = sklearn.metrics.log_loss(df_test["outcome"], xc)
    auc = sklearn.metrics.roc_auc_score(df_test["outcome"], xc)
    st.write("Test results")
    st.write(f"Brier: {brier}")
    st.write(f"Logloss: {logloss}")
    st.write(f"AUC: {auc}")
    return


def main():
    validate()
    return

    match, df_tracking, df_event = _get_preprocessed_data()

    df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()

    df_passes["xc"] = np.nan
    df_passes["AS"] = np.nan

    # df_passes["index"] = df_passes.index

    ### xC
    df_passes["xc"], _, _ = dangerous_accessible_space.get_expected_pass_completion(
        df_passes, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
        event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
        event_player_col="tracking_player_id",
    )

    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")
    st.write("-------")

    df_passes = df_passes.iloc[6:30]

    ### AS
    df_tracking = df_tracking[df_tracking["frame"].isin(df_passes["td_frame"])]
    df_tracking["AS"], df_tracking["result_index"], simulation_result = dangerous_accessible_space.get_dangerous_accessible_space(
        df_tracking, tracking_frame_col="frame", tracking_player_col="player_id", tracking_team_col="team_id",
    )
    df_passes["AS"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["AS"].to_dict())
    df_passes["result_index"] = df_passes["td_frame"].map(df_tracking.set_index("frame")["result_index"].to_dict())

    # df_passes = df_passes.sort_values("xc", ascending=True)

    for i, (frame, row) in enumerate(df_passes.iterrows()):
        plt.figure()
        fig, ax = databallpy.visualize.plot_soccer_pitch(pitch_color="white")
        databallpy.visualize.plot_tracking_data(
            match,
            row["td_frame"],
            team_colors=["blue", "red"],
            title=f"Pass completion: {row['outcome']}",
            add_velocities=True,
            variable_of_interest=f"AS={row['AS']:.0f} m^2, xC={row['xc']:.1%}",
            # variable_of_interest=f"AS={row['AS']:.0f} m^2",
            ax=ax,
        )
        team_color = "blue" if row["team_id"] == match.home_team_id else "red"
        def_team_color = "red" if row["team_id"] == match.home_team_id else "blue"
        plt.arrow(
            row["start_x"], row["start_y"], row["end_x"] - row["start_x"], row["end_y"] - row["start_y"], head_width=1,
            head_length=1, fc=team_color, ec=team_color
        )

        try:
            fig = plot_expected_completion_surface(
                simulation_result, row["result_index"], plot_type_off="poss", plot_type_def="poss", color_off=team_color,
                color_def=def_team_color, plot_gridpoints=True
            )
        except NameError as e:
            pass

        st.write(fig)
        plt.close(fig)

        if i > 30:
            break

    st.write(fig)

    profiler.stop()


if __name__ == '__main__':
    main()
