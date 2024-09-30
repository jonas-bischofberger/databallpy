import databallpy.features
import databallpy
import streamlit as st
import collections
import math
import numpy as np
import scipy
import time
from typing import Literal


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


def simulate_passes(
    PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
    BALL_POS,  # F x 2[x, y], ball positions
    phi_grid,  # F x PHI, pass angles
    v0_grid,  # F x V0, pass speeds
    passer_team,  # F, team of passers
    team_list,  # P, player teams
):
    ### 1. Calculate ball trajectory
    # 1.1 Calculate spatial grid
    start_location_offset = 0.001
    time_offset_ball = 0.0
    radial_gridsize = 5
    D_BALL_SIM = np.linspace(start=start_location_offset, stop=150, num=math.ceil(150 / radial_gridsize))  # T
    st.write("D_BALL_SIM", D_BALL_SIM.shape)
    st.write(D_BALL_SIM)

    def time_to_arrive_1d(x, v, x_target):
        return (x_target - x) / v

    # 1.2 Calculate temporal grid
    T_BALL_SIM = time_to_arrive_1d(
        x=D_BALL_SIM[0], v=v0_grid[:, :, np.newaxis], x_target=D_BALL_SIM[np.newaxis, np.newaxis, :],
    )  # F x V0 x T
    T_BALL_SIM += time_offset_ball
    st.write("T_BALL_SIM", T_BALL_SIM.shape)
    st.write(T_BALL_SIM[:, 0, :])

    def simulate_position(x, y, vx, vy, dt):
        return {
            "x": x + vx * dt,
            "y": y + vy * dt,
        }

    # 1.3 Calculate 2D points along ball trajectory
    st.write("BALL_POS", BALL_POS.shape)

    ball_x = BALL_POS[:, 0]  # F
    ball_y = BALL_POS[:, 1]  # F

    # cos_phi, sin_phi = np.cos(df_tracking_passes["phi"].values), np.sin(df_tracking_passes["phi"].values)  # F
    cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)  # F x PHI

    st.write("phi_grid", phi_grid.shape)
    st.write(phi_grid)

    st.write("cos_phi", cos_phi.shape, str(type(cos_phi)))
    st.write(cos_phi)
    st.write("v0_grid", v0_grid.shape, str(type(v0_grid)))
    st.write(v0_grid)
    v0x_ball = v0_grid[:, :, np.newaxis] * cos_phi[:, np.newaxis]  # F x V0 x PHI
    v0y_ball = v0_grid[:, :, np.newaxis] * sin_phi[:, np.newaxis]  # F x V0 x PHI
    st.write("v0y_ball", v0y_ball.shape)

    BALL_SIM = simulate_position(  # F x PHI x T
        x=ball_x[:, np.newaxis, np.newaxis],  # F
        y=ball_y[:, np.newaxis, np.newaxis],  # F
        vx=v0x_ball[:, 0, :, np.newaxis],  # F x V0 x PHI, positional grid is independent of V0!
        vy=v0y_ball[:, 0, :, np.newaxis],  # F x V0 x PHI
        dt=T_BALL_SIM[:, 0, np.newaxis, :]  # F x V0 x T
    )
    X_BALL_SIM = BALL_SIM["x"]  # F x PHI x T
    Y_BALL_SIM = BALL_SIM["y"]  # F x PHI x T
    st.write("X_BALL_SIM", X_BALL_SIM.shape)
    st.write(X_BALL_SIM[:, 0, :])
    st.write("Y_BALL_SIM", Y_BALL_SIM.shape)
    st.write(Y_BALL_SIM[:, 0, :])

    ### 2 Calculate player interception rates
    seconds_to_intercept = 0.5
    b0 = 0
    b1 = -5

    def time_to_arrive(x, y, vx, vy, x_target, y_target):
        return np.hypot(x_target - x, y_target - y) / np.hypot(vx, vy)

    # 2.1 Calculate time to arrive for each player
    st.write("PLAYER_POS", PLAYER_POS.shape)

    TTA_PLAYERS = time_to_arrive(  # F x P x PHI x T
        x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
        y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
        vx=PLAYER_POS[:, :, 2][:, :, np.newaxis, np.newaxis],
        vy=PLAYER_POS[:, :, 3][:, :, np.newaxis, np.newaxis],
        x_target=X_BALL_SIM[:, np.newaxis, :, :],
        y_target=Y_BALL_SIM[:, np.newaxis, :, :],
    )
    TTA_PLAYERS = np.nan_to_num(TTA_PLAYERS, nan=np.inf)  # Handle players not participating in the game by setting their TTA to infinity
    st.write("TTA_PLAYERS", TTA_PLAYERS.shape)
    st.write(TTA_PLAYERS[0, :, 0, :])

    # 2.2 Transform time to arrive into interception rates
    def sigmoid(x):
        return 0.5 * (x / (1 + np.abs(x)) + 1)

    X = TTA_PLAYERS[:, :, np.newaxis, :, :] - T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x P x PHI x T - F x PHI x T = F x P x V0 x PHI x T
    X[:] = b0 + b1 * X  # 1 + 1 * F x P x V0 x PHI x T = F x P x V0 x PHI x T
    X[:] = sigmoid(X)
    X = np.nan_to_num(X, nan=0)  # F x P x V0 x PHI x T

    ar_time = X / seconds_to_intercept  # F x P x V0 x PHI x T, interception rate / DR[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    st.write("ar_time", ar_time.shape)
    st.write(ar_time[0, :, 0, 0, :])

    ## 3. Use interception rates to calculate probabilities
    # 3.1 Sums of interception rates over players
    sum_ar = np.nansum(ar_time, axis=1)

    # player_list = np.array(player_list)
    # team_list = np.array(team_list)

    # st.write("team_list", team_list.shape)
    # st.write(team_list)

    # poss-specific
    player_is_attacking = team_list[np.newaxis, :] == passer_team[:, np.newaxis]  # F x P
    st.write("player_is_attacking", player_is_attacking.shape)
    st.write(player_is_attacking)

    x = np.where(player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    y = np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    sum_ar_att = np.nansum(x, axis=1)  # F x V0 x PHI x T
    sum_ar_def = np.nansum(y, axis=1)  # F x V0 x PHI x T

    st.write("sum_ar_att", sum_ar_att.shape)
    st.write(sum_ar_att[0, 0, 0, :])
    st.write("sum_ar_def", sum_ar_def.shape)
    st.write(sum_ar_def[0, 0, 0, :])

    def integrate_trapezoid(y, x):
        return scipy.integrate.cumulative_trapezoid(y=y, x=x, initial=0, axis=-1)  # * radial_gridsize # F x V0 x PHI x T

    # poss-specific
    ### Comment in to use Simpson integration, DO NOT USE until properly vectorized... it's a major bottleneck and probably not better than trapezoid
    int_sum_ar = integrate_trapezoid(y=sum_ar, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_att = integrate_trapezoid(y=sum_ar_att, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_def = integrate_trapezoid(y=sum_ar_def, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T

    st.write("int_sum_ar", int_sum_ar.shape)
    st.write(int_sum_ar[0, 0, 0, :])
    st.write("int_sum_ar_att", int_sum_ar_att.shape)
    st.write(int_sum_ar_att[0, 0, 0, :])
    st.write("int_sum_ar_def", int_sum_ar_def.shape)
    st.write(int_sum_ar_def[0, 0, 0, :])

    # Cumulative probabilities from integrals
    p0_cum = np.exp(-int_sum_ar) #if "prob" in ptypes else None  # F x V0 x PHI x T, cumulative probability that no one intercepted
    p0_cum_only_att = np.exp(-int_sum_ar_att) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_cum_only_def = np.exp(-int_sum_ar_def) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_only_opp = np.where(
        player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis],
        p0_cum_only_def[:, np.newaxis, :, :, :], p0_cum_only_att[:, np.newaxis, :, :, :]
    ) #if "poss" in ptypes else None  # F x P x V0 x PHI x T

    st.write("p0_cum", p0_cum.shape)
    st.write(p0_cum[0, 0, 0, :])
    st.write("p0_cum_only_att", p0_cum_only_att.shape)
    st.write(p0_cum_only_att[0, 0, 0, :])

    # Individual probability densities
    pr_prob = p0_cum[:, np.newaxis, :, :, :] * ar_time  # if "prob" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum = integrate_trapezoid(  # F x P x V0 x PHI x T, cumulative probability that player P intercepted
        y=pr_prob,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    st.write("pr_prob", pr_prob.shape)

    pr_poss = p0_only_opp * ar_time  # if "poss" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_poss = integrate_trapezoid(  # F x P x V0 x PHI x T
        y=pr_poss,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    st.write("pr_poss", pr_poss.shape)

    # Aggregate over v0
    # TODO use probability distribution. Bei prob auch phi-aggregation nötig -> ggf komplizierte Kombinationen...
    dpr_over_dx_vagg_prob = np.average(pr_prob, axis=2)  # if "prob" in ptypes else None  # F x P x PHI x T
    dpr_over_dx_vagg_poss = np.max(pr_poss, axis=2)  # if "poss" in ptypes else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269

    p0_cum_vagg = np.mean(p0_cum, axis=1)  # if add_receiver else None  # F x PHI x T
    pr_cum_vagg = np.mean(pr_cum, axis=2)  # if add_receiver else None  # F x P x PHI x T
    pr_cum_poss_vagg = np.max(pr_cum_poss, axis=2)  # if add_receiver else None  # F x P x V0 x PHI x T -> F x P x V0 x PHI x T

#     if self.cap_pos:
    pr_cum_poss_vagg = np.minimum(pr_cum_poss_vagg, 1)

    pr_cum_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T

    # if self.cap_pos:
    pr_cum_poss_att = np.minimum(pr_cum_poss_att, 1)
    pr_cum_poss_def = np.minimum(pr_cum_poss_def, 1)
    st.write("pr_cum_att", pr_cum_att.shape)
    st.write("pr_cum_poss_att", pr_cum_poss_att.shape)

    pr_cum_final_att = pr_cum_att[:, :, -1] #if add_receiver and add_final else None  # F x PHI
    pr_cum_poss_final_att = pr_cum_poss_att[:, :, -1] #if add_receiver and add_final else None
    pr_cum_final_def = pr_cum_def[:, :, -1] #if add_receiver and add_final else None  # F x PHI
    pr_cum_poss_final_def = pr_cum_poss_def[:, :, -1] #if add_receiver and add_final else None
    st.write("pr_cum_final_att", pr_cum_final_att.shape)
    st.write("pr_cum_poss_final_att", pr_cum_poss_final_att.shape)

    Result = collections.namedtuple("Result", [
        "pr_cum_att",
        "pr_cum_poss_att",
    ])
    result = Result(pr_cum_att, pr_cum_poss_att)

    return result


def get_expected_pass_completion(match):
    ed = match.event_data
    df = match.passes_df
    pe = match.pass_events
    td = match.tracking_data
    td = td.merge(ed[[col for col in ed.columns if col not in td.columns or col == "event_id"]], on="event_id", how="left")

    # check that no columns are duplicates
    assert len(set(td.columns)) == len(td.columns)

    x_cols = [col for col in match.tracking_data.columns if col.endswith("x")]
    y_cols = [col for col in match.tracking_data.columns if col.endswith("y")]

    # Normalize coordinates
    tracking_player_ids = match.home_players_column_ids() + match.away_players_column_ids() + ["ball"]
    position_cols = [f"{tracking_player_id}_{coord}" for tracking_player_id in tracking_player_ids for coord in ["x", "y", "vx", "vy"]] + ["start_x", "start_y", "end_x", "end_y"]
    i_away_possession = td["ball_possession"] == "away"
    td.loc[i_away_possession, position_cols] = -td.loc[i_away_possession, position_cols]

    td = td.rename(columns=lambda c: f"{c}_norm" if c in position_cols else c)
    position_cols_norm = [f"{col}_norm" for col in position_cols]

    st.write("ed")
    st.write(ed)
    df_passes = ed[ed["databallpy_event"] == "pass"]

    teamid2team = {match.home_team_id: "home", match.away_team_id: "away"}
    td["team"] = td["team_id"].map(teamid2team)

    # match.passes_df or the match.pass_events

    df_tracking = td

    df_tracking["event_id"] = df_tracking["event_id"].replace(-999, np.nan)
    df_tracking["pass_id"] = df_tracking["event_id"].where(df_tracking["databallpy_event"] == "pass", np.nan)
    df_tracking_passes = df_tracking[df_tracking["databallpy_event"] == "pass"]

    st.write("df_tracking_passes", df_tracking_passes.shape)
    st.write(df_tracking_passes.head(500))

    ### 1. Prepare data
    # 1.1 Extract player positions
    x_cols = [col for col in df_tracking_passes.columns if col.endswith("_x_norm") and not (col.startswith("start") or col.startswith("end"))]
    x_cols_players = [col for col in x_cols if "ball" not in col]

    coordinates = ["_x_norm", "_y_norm", "_vx_norm", "_vy_norm"]

    df_coords = df_tracking_passes[[f"{x_col.replace('_x_norm', coord)}" for x_col in x_cols_players for coord in coordinates]]

    st.write("df_coords")
    st.write(df_coords)

    F = df_coords.shape[0]  # number of frames
    C = len(coordinates)
    P = df_coords.shape[1] // len(coordinates)
    PLAYER_POS = df_coords.values.reshape(F, P, C)#.transpose(1, 0, 2)  # F x P x C
    st.write("PLAYER_POS", PLAYER_POS.shape)
    st.write(PLAYER_POS[0, :, :])

    # 1.2 Extract ball position
    BALL_POS = df_tracking_passes[[f"ball{coord}" for coord in coordinates]].values  # F x C
    st.write("BALL_POS", BALL_POS.shape)
    st.write(BALL_POS)

    # 1.3 Extract v0 as mean ball_velocity of the first N frames after the pass
    n_frames_after_pass_for_v0 = 5
    fallback_v0 = 10

    df_tracking_passes["pass_nr"] = df_tracking_passes.index
    index = [[idx + i for i in range(n_frames_after_pass_for_v0)] for idx in df_tracking_passes.index]
    index = [item for sublist in index for item in sublist]
    df_tracking_v0 = df_tracking.loc[index]
    df_tracking_v0["related_pass_id"] = df_tracking_v0["pass_id"].ffill()
    dfg_v0 = df_tracking_v0.groupby("related_pass_id")["ball_velocity"].mean()
    df_tracking_passes["v0"] = df_tracking_passes["pass_id"].map(dfg_v0)

    df_tracking_passes["v0"] = df_tracking_passes["v0"].fillna(fallback_v0)  # Set a reasonable default if no ball data was available during the first N frames

    v0_grid = df_tracking_passes["v0"].values[:, np.newaxis]  # F x V0

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

    player_list = [col.split("_x_norm")[0] for col in df_tracking_passes.columns if col.endswith("_x_norm") and not (col.startswith("start") or col.startswith("end") or "ball" in col)]  # P
    st.write("player_list", len(player_list))
    st.write(player_list)

    team_list = np.array(["home" if "home" in player else "away" for player in player_list])  # P
    player_list = np.array(player_list)  # P

    simulation_result = simulate_passes(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list)

    st.write("simulation_result")
    st.write(simulation_result)

    p = simulation_result.pr_cum_poss_att  # F x PHI x T
    xc = simulation_result.pr_cum_poss_att[:, 0, -1]  # F
    df_tracking_passes["xC"] = xc
    st.write("p.shape")
    st.write(p.shape)

    st.write("df_tracking_passes")
    st.write(df_tracking_passes)

    import sklearn.metrics

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

    st.write("xc", xc)

    # import databallpy.visualize

    for pass_index, p4ss in df_tracking_passes.iterrows():
        idx = pass_index
        import databallpy.visualize
        fig, ax = databallpy.visualize.plot_soccer_pitch(field_dimen=match.pitch_dimensions, pitch_color="white")
        st.write("match")
        st.write(match)
        print(match)
        print(type(match))
        fig, ax = databallpy.visualize.plot_tracking_data(
            match,
            idx,
            fig=fig,
            ax=ax,
            events=["pass"],
            title="First pass after the kick-off",
            add_velocities=True,
            variable_of_interest=df_tracking_passes.loc[idx, "xC"],
        )
        st.write(fig)
        break

    return df_tracking_passes


def main():
    match = get_preprocessed_data()

    df = match.event_data
    st.write("df")
    st.write(df)
    st.write(df["outcome"].mean())
    st.write(df["metrica_event"].unique())

    st.write("match.event_data")
    st.write(match.event_data)

    df_passes = get_expected_pass_completion(match)
    st.write("result df_passes")
    st.write(df_passes)


if __name__ == '__main__':
    main()
