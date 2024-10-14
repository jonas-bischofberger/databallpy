import json
import functools

import streamlit_profiler
import tqdm
import kloppy.metrica

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

    df_tracking = df_tracking

    df_events = match.event_data
    df_events["tracking_player_id"] = df_events["player_id"].map(match.player_id_to_column_id)

    return match, df_tracking, df_events


def validate_multiple_matches(
    dfs_tracking, dfs_passes, n_steps=100, training_size=0.75,

    outcome_col="success"
):
    random_state = 1893

    dfs_training = []
    dfs_test = []
    for dataset_nr, df_passes in enumerate(dfs_passes):
        dataset_nr_col = dangerous_accessible_space.interface._get_unused_column_name(df_passes, "dataset_nr")
        df_passes[dataset_nr_col] = dataset_nr
        df_training, df_test = sklearn.model_selection.train_test_split(df_passes, stratify=df_passes[outcome_col], train_size=training_size, random_state=random_state)
        # df_training, df_test = df_passes, df_passes
        dfs_training.append(df_training)
        dfs_test.append(df_test)

    df_training = pd.concat(dfs_training)
    df_test = pd.concat(dfs_test)

    st.write("Number of training passes", len(df_training), df_training[outcome_col].mean())
    st.write("Number of test passes", len(df_test), df_test[outcome_col].mean())

    average_accuracy_training = df_training[outcome_col].mean()
    st.write("Average Accuracy training", average_accuracy_training)
    baseline_brier = sklearn.metrics.brier_score_loss(df_test[outcome_col], [average_accuracy_training] * len(df_test))
    st.write("Baseline brier (test)", baseline_brier)
    try:
        baseline_logloss = sklearn.metrics.log_loss(df_test[outcome_col], [average_accuracy_training] * len(df_test))
    except ValueError:
        baseline_logloss = np.nan
    st.write("Baseline logloss", baseline_logloss)
    try:
        baseline_auc = sklearn.metrics.roc_auc_score(df_test[outcome_col], [average_accuracy_training] * len(df_test))
    except ValueError:
        baseline_auc = np.nan

    st.write("Baseline AUC", baseline_auc)

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

        data_simres = {
            "xc": [],
            "success": [],
        }
        for dataset_nr, df_training_passes in df_training.groupby("dataset_nr"):
            df_tracking = dfs_tracking[dataset_nr]
            # st.write("df_tracking")
            # st.write(df_tracking[["frame_id", "player_id", "team_id", "x", "y", "vx", "vy", "v"]].head(500))
            # st.write("df_training_passes")
            # st.write(df_training_passes[["frame_id", "player_id", "team_id", "coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y"]])
            df_training_passes = df_training_passes.sort_values("frame_id")
            xc, _, _ = dangerous_accessible_space.get_expected_pass_completion(
                df_training_passes, df_tracking, event_frame_col="frame_id", tracking_frame_col="frame_id", event_start_x_col="coordinates_x",
                event_start_y_col="coordinates_y", event_end_x_col="end_coordinates_x", event_end_y_col="end_coordinates_y",
                event_team_col="team_id",
                event_player_col="player_id", tracking_player_col="player_id", tracking_team_col="team_id", ball_tracking_player_id="ball",
                tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col="v",

                n_frames_after_pass_for_v0=5, fallback_v0=10,

                **random_paramter_assignment,

                # tracking_team_col="team_id", ball_tracking_player_id="ball",
                #     n_frames_after_pass_for_v0=5, fallback_v0=10, tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx",
                #     tracking_vy_col="vy", tracking_v_col=None, event_start_x_col="x", event_start_y_col="y",
                #     event_end_x_col="x_target", event_end_y_col="y_target", event_team_col="team_id", event_player_col="",
                #     outcome_col="success",
            )
            data_simres["xc"].extend(xc)
            data_simres["success"].extend(df_training_passes[outcome_col].tolist())

        df_simres = pd.DataFrame(data_simres)
        brier = sklearn.metrics.brier_score_loss(df_simres["success"], df_simres["xc"])
        try:
            logloss = sklearn.metrics.log_loss(df_simres["success"], df_simres["xc"])
        except ValueError:
            logloss = np.nan
        try:
            auc = sklearn.metrics.roc_auc_score(df_simres["success"], df_simres["xc"])
        except ValueError:
            auc = np.nan

        data["brier"].append(brier)
        data["logloss"].append(logloss)
        data["auc"].append(auc)
        data["parameters"].append(random_paramter_assignment)

        display_df.write(pd.DataFrame(data).sort_values("logloss"), ascending=True)

    df_training_results = pd.DataFrame(data)
    # st.write("df_training_results")
    # st.write(df_training_results)

    best_index = df_training_results["logloss"].idxmin()
    best_parameters = df_training_results["parameters"][best_index]
    # st.write("Best parameters")
    # st.write(best_parameters)

    data_simres = {
        "xc": [],
        "success": [],
    }
    for dataset_nr, df_test_passes in df_test.groupby("dataset_nr"):
        df_tracking = dfs_tracking[dataset_nr]
        # st.write("df_tracking")
        # st.write(df_tracking[["frame_id", "player_id", "team_id", "x", "y", "vx", "vy", "v"]].head(500))
        # st.write("df_training_passes")
        # st.write(df_training_passes[["frame_id", "player_id", "team_id", "coordinates_x", "coordinates_y", "end_coordinates_x", "end_coordinates_y"]])
        df_training_passes = df_test_passes.sort_values("frame_id")
        xc, _, _ = dangerous_accessible_space.get_expected_pass_completion(
            df_test_passes, df_tracking, event_frame_col="frame_id", tracking_frame_col="frame_id",
            event_start_x_col="coordinates_x",
            event_start_y_col="coordinates_y", event_end_x_col="end_coordinates_x", event_end_y_col="end_coordinates_y",
            event_team_col="team_id",
            event_player_col="player_id", tracking_player_col="player_id", tracking_team_col="team_id",
            ball_tracking_player_id="ball",
            tracking_x_col="x", tracking_y_col="y", tracking_vx_col="vx", tracking_vy_col="vy", tracking_v_col="v",

            n_frames_after_pass_for_v0=5, fallback_v0=10,

            **best_parameters,
        )
        data_simres["xc"].extend(xc)
        data_simres["success"].extend(df_training_passes[outcome_col].tolist())
    df_simres = pd.DataFrame(data_simres)
    brier = sklearn.metrics.brier_score_loss(df_simres["success"], df_simres["xc"])
    logloss = sklearn.metrics.log_loss(df_simres["success"], df_simres["xc"])
    auc = sklearn.metrics.roc_auc_score(df_simres["success"], df_simres["xc"])

    # brier = sklearn.metrics.brier_score_loss(df_test[outcome_col], df_test["xc"])
    # logloss = sklearn.metrics.log_loss(df_test[outcome_col], df_test["xc"])
    # auc = sklearn.metrics.roc_auc_score(df_test[outcome_col], df_test["xc"])
    st.write("Test results")
    st.write(f"Brier: {brier}")
    st.write(f"Logloss: {logloss}")
    st.write(f"AUC: {auc}")

    brier_skill_score = 1 - brier / baseline_brier
    st.write(f"Brier skill score: {brier_skill_score}")

    return


def validate(n_steps=100, training_size=0.99):
    _, df_tracking, df_event = _get_preprocessed_data()
    df_passes = df_event[df_event["databallpy_event"] == "pass"].reset_index()

    random_state = 1893
    df_training, df_test = sklearn.model_selection.train_test_split(df_passes, train_size=training_size, random_state=random_state)

    # df_training = df_passes
    # df_test = df_passes

    average_accuracy = df_training["outcome_col"].mean()
    baseline_brier = sklearn.metrics.brier_score_loss(df_training["outcome"], [average_accuracy] * len(df_training))
    st.write("Baseline brier", baseline_brier)
    try:
        baseline_logloss = sklearn.metrics.log_loss(df_training["outcome"], [average_accuracy] * len(df_training))
    except ValueError:
        baseline_logloss = np.nan
    st.write("Baseline logloss", baseline_logloss)
    baseline_auc = sklearn.metrics.roc_auc_score(df_training["outcome"], [average_accuracy] * len(df_training))
    st.write("Baseline AUC", baseline_auc)

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


@st.cache_resource
def get_kloppy_data():
    datasets = []
    dfs_event = []
    st.write(" ")
    st.write(" ")
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    for dataset_nr in [1, 2, 3]:
        progress_bar_text.text(f"Loading dataset {dataset_nr}")
        # dataset = kloppy.metrica.load_tracking_csv(
        #     home_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Home_Team.csv",
        #     away_data=f"https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawTrackingData_Away_Team.csv",
        #     # sample_rate=1 / 5,
        #     # limit=100,
        #     coordinates="secondspectrum"
        # )
        # df_events1 = pd.read_csv(f"https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv")
        # df_passes1 = df_events1[df_events1["Type"] == "PASS"]
        dataset = kloppy.metrica.load_open_data(dataset_nr)#, limit=100)
        # datasets.append((dataset.to_df(), df_passes1))
        df_tracking = dataset.to_df()
        df_tracking[[col for col in df_tracking.columns if col.endswith("_x")]] = (df_tracking[[col for col in df_tracking.columns if col.endswith("_x")]].astype(float) - 0.5) * 105
        df_tracking[[col for col in df_tracking.columns if col.endswith("_y")]] = (df_tracking[[col for col in df_tracking.columns if col.endswith("_y")]].astype(float) - 0.5) * 68

        df_tracking = df_tracking.drop(columns=[col for col in df_tracking.columns if col.endswith("_d") or col.endswith("_s")])

        players = [col.replace("_x", "") for col in df_tracking.columns if col.endswith("_x")]
        x_cols = [f"{player}_x" for player in players]
        y_cols = [f"{player}_y" for player in players]
        vx_cols = [f"{player}_vx" for player in players]
        vy_cols = [f"{player}_vy" for player in players]
        v_cols = [f"{player}_velocity" for player in players]
        frame_col = "frame_id"

        # dt = df_tracking["timestamp"].diff().mean()

        # df_tracking["ball_vx"] = df_tracking["ball_x"].diff() / df_tracking["timestamp"].dt.total_seconds().diff()
        # df_tracking["ball_vy"] = df_tracking["ball_y"].diff() / df_tracking["timestamp"].dt.total_seconds().diff()
        # df_tracking["ball_velocity"] = np.sqrt(df_tracking["ball_vx"]**2 + df_tracking["ball_vy"]**2)
        for player in players:
            df_tracking[f"{player}_x"] = df_tracking[f"{player}_x"].astype(float)
            xdiff = df_tracking[f"{player}_x"].diff().fillna(method="bfill")
            xdiff2 = -df_tracking[f"{player}_x"].diff(periods=-1).fillna(method="ffill")
            tdiff = df_tracking["timestamp"].diff().dt.total_seconds().fillna(method="bfill")
            tdiff2 = -df_tracking["timestamp"].diff(periods=-1).dt.total_seconds().fillna(method="ffill")
            vx = (xdiff + xdiff2) / (tdiff + tdiff2)
            df_tracking[f"{player}_vx"] = vx

            df_tracking[f"{player}_y"] = df_tracking[f"{player}_y"].astype(float)
            ydiff = df_tracking[f"{player}_y"].diff().fillna(method="bfill")
            ydiff2 = -df_tracking[f"{player}_y"].diff(periods=-1).fillna(method="ffill")
            vy = (ydiff + ydiff2) #/ (tdiff + tdiff2)
            df_tracking[f"{player}_vy"] = vy
            df_tracking[f"{player}_velocity"] = np.sqrt(vx**2 + vy**2)

            i_nan_x = df_tracking[f"{player}_x"].isna()
            df_tracking.loc[i_nan_x, f"{player}_vx"] = np.nan
            i_nan_y = df_tracking[f"{player}_y"].isna()
            df_tracking.loc[i_nan_y, f"{player}_vy"] = np.nan
            df_tracking.loc[i_nan_x | i_nan_y, f"{player}_velocity"] = np.nan

        df_events = get_kloppy_events(dataset_nr).copy()

        player_to_team = {}
        if dataset_nr in [1, 2]:
            for player in players:
                if "home" in player:
                    player_to_team[player] = "Home"
                elif "away" in player:
                    player_to_team[player] = "Away"
                else:
                    player_to_team[player] = None
        else:
            player_to_team = df_events[['player_id', 'team_id']].set_index('player_id')['team_id'].to_dict()

        df_tracking_obj = dangerous_accessible_space.per_object_frameify_tracking_data(df_tracking, frame_col, x_cols, y_cols, vx_cols, vy_cols, players, player_to_team, v_cols=v_cols)

        # get ball control
        fr2control = df_events.set_index("frame_id")["team_id"].to_dict()
        df_tracking_obj["ball_possession"] = df_tracking_obj["frame_id"].map(fr2control)
        df_tracking_obj = df_tracking_obj.sort_values("frame_id")
        df_tracking_obj["ball_possession"] = df_tracking_obj["ball_possession"].ffill()

        datasets.append(df_tracking_obj)

        dfs_event.append(df_events)

        progress_bar.progress(dataset_nr / 3)



    # dataset3 = kloppy.metrica.load_open_data(3)

    return datasets, dfs_event



@st.cache_resource
def get_kloppy_events(dataset_nr):
    if dataset_nr in [1, 2]:
        df = pd.read_csv(f"C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_{dataset_nr}/Sample_Game_{dataset_nr}_RawEventsData.csv")
        df["body_part_type"] = df["Subtype"].where(df["Subtype"].isin(["HEAD"]), None)
        df["set_piece_type"] = df["Subtype"].where(df["Subtype"].isin(["THROW IN", "GOAL KICK", "FREE KICK", "CORNER KICK"]), None).map(lambda x: x.replace(" ", "_") if x is not None else None)
        df["Type"] = df["Type"].str.replace(" ", "_")
        df["Start X"] = (df["Start X"] - 0.5) * 105
        df["Start Y"] = -(df["Start Y"] - 0.5) * 68
        df["End X"] = (df["End X"] - 0.5) * 105
        df["End Y"] = -(df["End Y"] - 0.5) * 68
        df = df.rename(columns={
            "Type": "event_type",
            "Period": "period_id",
            "Team": "team_id",
            "From": "player_id",
            "To": "receiver_player_id",
            "Start X": "coordinates_x",
            "Start Y": "coordinates_y",
            "End X": "end_coordinates_x",
            "End Y": "end_coordinates_y",
            "Start Frame": "frame_id",
            "End Frame": "end_frame_id",
        })
        player_id_to_column_id = {}
        column_id_to_team_id = {}
        for team_id in df["team_id"].unique():
            df_players = df[df["team_id"] == team_id]
            team_player_ids = set(df_players["player_id"].dropna().tolist() + df_players["receiver_player_id"].dropna().tolist())
            player_id_to_column_id.update({player_id: f"{team_id.lower().strip()}_{player_id.replace('Player', '').strip()}" for player_id in team_player_ids})
            column_id_to_team_id.update({player_id_to_column_id[player_id]: team_id for player_id in team_player_ids})

        df["player_id"] = df["player_id"].map(player_id_to_column_id)
        df["receiver_player_id"] = df["receiver_player_id"].map(player_id_to_column_id)
        df["receiver_team_id"] = df["receiver_player_id"].map(column_id_to_team_id)

        df["tmp_next_player"] = df["player_id"].shift(-1)
        df["tmp_next_team"] = df["team_id"].shift(-1)
        df["tmp_receiver_player"] = df["receiver_player_id"].where(df["receiver_player_id"].notna(), df["tmp_next_player"])
        df["tmp_receiver_team"] = df["tmp_receiver_player"].map(column_id_to_team_id)

        df["success"] = df["tmp_receiver_team"] == df["team_id"]

        df["is_pass"] = (df["event_type"].isin(["PASS", "BALL_LOST", "BALL_OUT"])) \
                        & (~df["Subtype"].isin(["CLEARANCE", "HEAD-CLEARANCE", "HEAD-INTERCEPTION-CLEARANCE"])) \
                        & (df["frame_id"] != df["end_frame_id"])

        df["is_high"] = df["Subtype"].isin([
            "CROSS",
            # "CLEARANCE",
            "CROSS-INTERCEPTION",
            # "HEAD-CLEARANCE",
            # "HEAD-INTERCEPTION-CLEARANCE"
        ])

        #     df_passes["xc"], _, _ = dangerous_accessible_space.get_expected_pass_completion(
        #         df_passes, df_tracking, event_frame_col="td_frame", tracking_frame_col="frame", event_start_x_col="start_x",
        #         event_start_y_col="start_y", event_end_x_col="end_x", event_end_y_col="end_y",
        #         event_player_col="tracking_player_id",
        #     )

        return df.drop(columns=["tmp_next_player", "tmp_next_team", "tmp_receiver_player", "tmp_receiver_team"])
    else:
        # dataset = kloppy.metrica.load_event(
        #     event_data="C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_events.json",
        #     # meta_data="https://raw.githubusercontent.com/metrica-sports/sample-data/refs/heads/master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
        #     meta_data="C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_metadata.xml",
        #     coordinates="secondspectrum",
        # )
        json_data = json.load(open("C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_events.json"))
        import xmltodict

        df = pd.json_normalize(json_data["data"])

        expanded_df = pd.DataFrame(df['subtypes'].apply(pd.Series))
        expanded_df.columns = [f'subtypes.{col}' for col in expanded_df.columns]

        new_dfs = []
        for expanded_col in expanded_df.columns:
            expanded_df2 = pd.json_normalize(expanded_df[expanded_col])
            expanded_df2.columns = [f'{expanded_col}.{col}' for col in expanded_df2.columns]
            new_dfs.append(expanded_df2)

        expanded_df = pd.concat(new_dfs, axis=1)

        df = pd.concat([df, expanded_df], axis=1)

        i_subtypes_nan = ~df["subtypes.name"].isna()
        i_subtypes_0_nan = ~df["subtypes.0.name"].isna()

        # check if the true's are mutually exclusive
        assert not (i_subtypes_nan & i_subtypes_0_nan).any()

        df.loc[i_subtypes_nan, "subtypes.0.name"] = df.loc[i_subtypes_nan, "subtypes.name"]
        df.loc[i_subtypes_nan, "subtypes.0.id"] = df.loc[i_subtypes_nan, "subtypes.id"]
        df = df.drop(columns=["subtypes.name", "subtypes.id", "subtypes"])
        subtype_cols = [col for col in df.columns if col.startswith("subtypes.") and col.endswith("name")]

        player2team = df[['from.id', 'team.id']].set_index('from.id')['team.id'].to_dict()
        df["receiver_team_id"] = df["to.id"].map(player2team)
        # df["tmp_next_player"] = df["player_id"].shift(-1)
        # df["tmp_next_team"] = df["team_id"].shift(-1)
        # df["tmp_receiver_player"] = df["receiver_player_id"].where(df["receiver_player_id"].notna(), df["tmp_next_player"])
        # df["tmp_receiver_team"] = df["tmp_receiver_player"].map(column_id_to_team_id)
        df["success"] = df["receiver_team_id"] == df["team.id"]

        df["success"] = df["success"].astype(bool)

        df["is_pass"] = (df["type.name"].isin(["PASS", "BALL LOST", "BALL OUT"])) \
                        & ~df[subtype_cols].isin(["CLEARANCE"]).any(axis=1) \
                        & (df["start.frame"] != df["end.frame"])

# df[df[['Name', 'Age']].isin(['Alice', 30]).any(axis=1)]
        df["is_high"] = df[subtype_cols].isin([
            "CROSS",
        ]).any(axis=1)

        df = df.rename(columns={
            "type.name": "event_type",
            "from.id": "player_id",
            "team.id": "team_id",
            "to.id": "receiver_player_id",
            "period": "period_id",
            "start.frame": "frame_id",
            "end.frame": "end_frame_id",
            "start.x": "coordinates_x",
            "start.y": "coordinates_y",
            "end.x": "end_coordinates_x",
            "end.y": "end_coordinates_y",
        }).drop(columns=[
            "to",
        ])
        df["coordinates_x"] = (df["coordinates_x"] - 0.5) * 105
        df["coordinates_y"] = (df["coordinates_y"] - 0.5) * 68
        df["end_coordinates_x"] = (df["end_coordinates_x"] - 0.5) * 105
        df["end_coordinates_y"] = (df["end_coordinates_y"] - 0.5) * 68

        meta_data = xmltodict.parse(open("C:/Users/Jonas/Desktop/ucloud/Arbeit/Spielanalyse/soccer-analytics/football1234/datasets/metrica/sample-data-master/data/Sample_Game_3/Sample_Game_3_metadata.xml").read())
        df_player = pd.json_normalize(meta_data, record_path=["main", "Metadata", "Players", "Player"])
        player2team = df_player[["@id", "@teamId"]].set_index("@id")["@teamId"].to_dict()
        df["team_id"] = df["player_id"].map(player2team)

        return df


def main():
    st.write(f"Getting kloppy data...")
    dfs_tracking, dfs_event = get_kloppy_data()

    # dfs_tracking = [dfs_tracking[1]]
    # dfs_event = [dfs_event[1]]

    # dfs_tracking = datasets
    # dfs_event = []
    dfs_passes = []
    for i, (df_tracking, df_events) in enumerate(zip(dfs_tracking, dfs_event)):
        df_events["player_id"] = df_events["player_id"].str.replace(" ", "")
        df_events["receiver_player_id"] = df_events["receiver_player_id"].str.replace(" ", "")

        ### Prepare data -> TODO put into other function
        dataset_nr = i+1
        st.write(f"### Dataset {dataset_nr}")
        # if dataset_nr == 1 or dataset_nr == 2:
        #     continue
        # df_tracking = dataset
        # st.write(f"Getting events...")
        # df_events = get_kloppy_events(dataset_nr)

        st.write("Pass %", f'{df_events[df_events["is_pass"]]["success"].mean():.2%}', f'Passes: {len(df_events[df_events["is_pass"]])}')

        st.write("df_tracking", df_tracking.shape)
        st.write(df_tracking.head())
        st.write("df_events", df_events.shape)
        st.write(df_events)

        ### Do validation with this data
        dfs_event.append(df_events)
        df_passes = df_events[(df_events["is_pass"]) & (~df_events["is_high"])]

        df_passes = df_passes.drop_duplicates(subset=["frame_id"])

        # st.write("df_passes", df_passes.shape)
        # st.write("df_passes_fr_unique", len(df_passes["frame_id"].unique()))

        duplicate_frames = df_passes["frame_id"].value_counts()
        duplicate_frames = duplicate_frames[duplicate_frames > 1]

        # st.write(df_passes)
        # dfs_passes.append(df_passes.iloc[125:126])
        dfs_passes.append(df_passes)

        for _, p4ss in df_passes.iloc[125:126].iterrows():
            plt.figure()
            plt.arrow(x=p4ss["coordinates_x"], y=p4ss["coordinates_y"], dx=p4ss["end_coordinates_x"] - p4ss["coordinates_x"], dy=p4ss["end_coordinates_y"] - p4ss["coordinates_y"], head_width=1, head_length=1, fc="blue", ec="blue")
            df_frame = df_tracking[df_tracking["frame_id"] == p4ss["frame_id"]]
            for team in df_frame["team_id"].unique():
                df_frame_team = df_frame[df_frame["team_id"] == team]
                x = df_frame_team["x"].tolist()
                y = df_frame_team["y"].tolist()
                plt.scatter(x, y, c="red" if team == p4ss["team_id"] else "blue")

            plt.plot([-52.5, 52.5], [-34, -34], c="black")
            plt.plot([-52.5, 52.5], [34, 34], c="black")
            plt.plot([-52.5, -52.5], [-34, 34], c="black")
            plt.plot([52.5, 52.5], [-34, 34], c="black")
            plt.title(f"Pass: {p4ss['success']}")
            st.write(plt.gcf())
            # plt.show()
            # st.stop()
            # break

    # validate()
    n_steps = st.slider("Number of simulations", 1, 2000, 1)
    validate_multiple_matches(dfs_tracking=dfs_tracking, dfs_passes=dfs_passes, outcome_col="success", n_steps=n_steps)
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
