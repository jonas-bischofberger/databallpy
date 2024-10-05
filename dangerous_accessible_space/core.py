import numpy as np
import streamlit as st
import math
import scipy.integrate
import collections


Result = collections.namedtuple("Result", [
    "p_cum_att",  # F x PHI x T
    "p_density_att",  # F x PHI x T
    "phi_grid",  # PHI
    "r_grid",  # T
    # "on_pitch_mask",  # F x PHI x T
    # "x0_grid",  # F
    # "y0_grid",  # F
    "x_grid",  # F x PHI x T
    "y_grid",  # F x PHI x T
])


def simulate_passes(
    PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
    BALL_POS,  # F x 2[x, y], ball positions
    phi_grid,  # F x PHI, pass angles
    v0_grid,  # F x V0, pass speeds
    passer_team,  # F, team of passers
    team_list,  # P, player teams
) -> Result:
    start_location_offset = 0
    time_offset_ball = 0.0
    radial_gridsize = 3
    seconds_to_intercept = 0.01
    b0 = 0
    b1 = -15

    ### 1. Calculate ball trajectory
    # 1.1 Calculate spatial grid
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
    st.write("T_BALL_SIM[fr=0, V0=0, T=:]", T_BALL_SIM.shape)
    st.write(T_BALL_SIM[0, 0, :])

    # 1.3 Calculate 2D points along ball trajectory
    st.write("BALL_POS", BALL_POS.shape)
    st.write(BALL_POS)
    st.write("np.cos(phi_grid)", np.cos(phi_grid).shape)

    # cos_phi, sin_phi = np.cos(df_tracking_passes["phi"].values), np.sin(df_tracking_passes["phi"].values)  # F
    cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)  # F x PHI

    # ball_x0 = BALL_POS[:, 0]
    X_BALL_SIM = BALL_POS[:, 0][:, np.newaxis, np.newaxis] + cos_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T
    # ball_y0 = BALL_POS[:, 1]
    Y_BALL_SIM = BALL_POS[:, 1][:, np.newaxis, np.newaxis] + sin_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T

    st.write("X_BALL_SIM", X_BALL_SIM.shape)
    st.write(X_BALL_SIM[0, :, :])
    st.write("Y_BALL_SIM", Y_BALL_SIM.shape)
    st.write(Y_BALL_SIM[0, :, :])

    # st.write("X_BALL_SIM", X_BALL_SIM.shape)
    # st.write(X_BALL_SIM[0, :, :])

    # st.write("phi_grid", phi_grid.shape)
    # st.write(phi_grid)

    # st.write("cos_phi", cos_phi.shape, str(type(cos_phi)))
    # st.write(cos_phi)
    # st.write("v0_grid", v0_grid.shape, str(type(v0_grid)))
    # st.write(v0_grid)

    # v0x_ball = v0_grid[:, :, np.newaxis] * cos_phi[:, np.newaxis]  # F x V0 x PHI
    # v0y_ball = v0_grid[:, :, np.newaxis] * sin_phi[:, np.newaxis]  # F x V0 x PHI

    # BALL_SIM = simulate_position(  # F x PHI x T
    #     x=ball_x,  # F x PHI x T
    #     y=ball_y,  # F x PHI x T
    #     vx=v0x_ball[:, 0, :, np.newaxis],  # F x V0 x PHI, positional grid is independent of V0!
    #     vy=v0y_ball[:, 0, :, np.newaxis],  # F x V0 x PHI
    #     dt=T_BALL_SIM[:, 0, np.newaxis, :]  # F x V0 x T
    # )
    # X_BALL_SIM = BALL_SIM["x"]  # F x PHI x T
    # Y_BALL_SIM = BALL_SIM["y"]  # F x PHI x T
    #
    # st.write("X_BALL_SIM", X_BALL_SIM.shape)
    # st.write(X_BALL_SIM[0, :, :])

    # st.stop()

    ### 2 Calculate player interception rates
    def time_to_arrive(x, y, vx, vy, x_target, y_target):
        # return np.hypot(x_target - x, y_target - y) / np.hypot(vx, vy)
        D = np.hypot(x_target - x, y_target - y)
        V = 4
        st.write("D", D.shape)
        st.write(D[0, :, 0, :])
        return D / V
        # return np.hypot(x_target - x, y_target - y) / np.hypot(vx, vy)

    st.write("PLAYER_POS A", PLAYER_POS.shape)
    st.write(PLAYER_POS[0, :, :])

    # 2.1 Calculate time to arrive for each player
    TTA_PLAYERS = time_to_arrive(  # F x P x PHI x T
        x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
        y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
        vx=PLAYER_POS[:, :, 2][:, :, np.newaxis, np.newaxis],
        vy=PLAYER_POS[:, :, 3][:, :, np.newaxis, np.newaxis],
        x_target=X_BALL_SIM[:, np.newaxis, :, :],
        y_target=Y_BALL_SIM[:, np.newaxis, :, :],
    )
    TTA_PLAYERS = np.nan_to_num(TTA_PLAYERS, nan=np.inf)  # Handle players not participating in the game by setting their TTA to infinity
    st.write("TTA_PLAYERS[fr=0, P=:, PHI=0, T=:]")
    st.write(TTA_PLAYERS[0, :, 0, :])

    # 2.2 Transform time to arrive into interception rates
    # def sigmoid(x):
    #     return 0.5 * (x / (1 + np.abs(x)) + 1)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    X = TTA_PLAYERS[:, :, np.newaxis, :, :] - T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x P x PHI x T - F x PHI x T = F x P x V0 x PHI x T

    st.write("X")
    st.write(X[0, :, 0, 0, :])

    X[:] = b0 + b1 * X  # 1 + 1 * F x P x V0 x PHI x T = F x P x V0 x PHI x T
    X[:] = sigmoid(X)
    X = np.nan_to_num(X, nan=0)  # F x P x V0 x PHI x T

    ar_time = X / seconds_to_intercept  # F x P x V0 x PHI x T, interception rate / DR[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    st.write("ar_time[fr=0, P=:, V0=0, PHI=0, T=:]")
    st.write(ar_time[0, :, 0, 0, :])

    ## 3. Use interception rates to calculate probabilities
    # 3.1 Sums of interception rates over players
    sum_ar = np.nansum(ar_time, axis=1)

    # player_list = np.array(player_list)
    # team_list = np.array(team_list)

    # poss-specific
    player_is_attacking = team_list[np.newaxis, :] == passer_team[:, np.newaxis]  # F x P

    x = np.where(player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    y = np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    sum_ar_att = np.nansum(x, axis=1)  # F x V0 x PHI x T
    sum_ar_def = np.nansum(y, axis=1)  # F x V0 x PHI x T

    def integrate_trapezoid(y, x):
        return scipy.integrate.cumulative_trapezoid(y=y, x=x, initial=0, axis=-1)  # * radial_gridsize # F x V0 x PHI x T

    # poss-specific
    int_sum_ar = integrate_trapezoid(y=sum_ar, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_att = integrate_trapezoid(y=sum_ar_att, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_def = integrate_trapezoid(y=sum_ar_def, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T

    # Cumulative probabilities from integrals
    p0_cum = np.exp(-int_sum_ar) #if "prob" in ptypes else None  # F x V0 x PHI x T, cumulative probability that no one intercepted
    p0_cum_only_att = np.exp(-int_sum_ar_att) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_cum_only_def = np.exp(-int_sum_ar_def) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_only_opp = np.where(
        player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis],
        p0_cum_only_def[:, np.newaxis, :, :, :], p0_cum_only_att[:, np.newaxis, :, :, :]
    ) #if "poss" in ptypes else None  # F x P x V0 x PHI x T

    # Individual probability densities
    pr_prob = p0_cum[:, np.newaxis, :, :, :] * ar_time  # if "prob" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum = integrate_trapezoid(  # F x P x V0 x PHI x T, cumulative probability that player P intercepted
        y=pr_prob,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    pr_poss = p0_only_opp * ar_time  # if "poss" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_poss = integrate_trapezoid(  # F x P x V0 x PHI x T
        y=pr_poss,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    # Aggregate over v0
    # TODO use probability distribution. Bei prob auch phi-aggregation nötig -> ggf komplizierte Kombinationen...
    dpr_over_dx_vagg_prob = np.average(pr_prob, axis=2)  # if "prob" in ptypes else None  # F x P x PHI x T
    dpr_over_dx_vagg_poss = np.max(pr_poss, axis=2)  # if "poss" in ptypes else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269

    p0_cum_vagg = np.mean(p0_cum, axis=1)  # if add_receiver else None  # F x PHI x T
    pr_cum_vagg = np.mean(pr_cum, axis=2)  # if add_receiver else None  # F x P x PHI x T
    pr_cum_poss_vagg = np.max(pr_cum_poss, axis=2)  # if add_receiver else None  # F x P x V0 x PHI x T -> F x P x V0 x PHI x T

    pr_cum_poss_vagg = np.minimum(pr_cum_poss_vagg, 1)

    dpr_over_dx_vagg_att_poss = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_poss, 0), axis=1)  #if add_receiver else None  # F x PHI x T
    dpr_over_dx_vagg_att_poss = np.minimum(dpr_over_dx_vagg_att_poss, 1/radial_gridsize)

    pr_cum_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T

    # if self.cap_pos:
    pr_cum_poss_att = np.minimum(pr_cum_poss_att, 1)
    pr_cum_poss_def = np.minimum(pr_cum_poss_def, 1)

    # on_pitch_mask = ~((X_BALL_SIM >= -52.5) & (X_BALL_SIM <= 52.5) & (Y_BALL_SIM >= -34) & (Y_BALL_SIM <= 34))  # F x PHI x T

    # st.write("X_BALL_SIM[fr=0]", X_BALL_SIM.shape)
    # st.write(X_BALL_SIM[0, :, :])
    # st.write("Y_BALL_SIM[fr=0]", Y_BALL_SIM.shape)
    # st.write(Y_BALL_SIM[0, :, :])
    # st.write("on_pitch_mask[fr=0]", on_pitch_mask.shape)
    # st.write(on_pitch_mask[0, :, :])
    # st.write("pr_cum_poss_att[fr=0]", pr_cum_poss_att.shape)
    # st.write(pr_cum_poss_att[0, :, :])

    x0_grid = X_BALL_SIM[:, 0, 0]  # F x T
    y0_grid = Y_BALL_SIM[:, 0, 0]  # F x T

    st.write("x0_grid", x0_grid.shape)
    st.write(x0_grid)

    st.write("BALL_POS", BALL_POS.shape)

    # result = Result(pr_cum_poss_att, phi_grid, D_BALL_SIM, on_pitch_mask, X_BALL_SIM[:, 0, 0], Y_BALL_SIM[:, 0, 0])
    result = Result(
        p_cum_att=pr_cum_poss_att,
        p_density_att=dpr_over_dx_vagg_att_poss,
        phi_grid=phi_grid, r_grid=D_BALL_SIM, x_grid=X_BALL_SIM, y_grid=Y_BALL_SIM
    )

    return result


def simulate_passes_chunked(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list) -> Result:
    F = PLAYER_POS.shape[0]
    chunk_size = 10
    i_chunks = np.arange(0, F, chunk_size)
    # i_chunks = np.arange(2000, 10000, chunk_size)

    full_result = None
    progress_bar = st.progress(0)
    progress_text = st.empty()
    for chunk_nr, i in enumerate(i_chunks):
        progress_text.text(f"Simulating chunk {chunk_nr + 1}/{len(i_chunks)}")
        progress_bar.progress(chunk_nr / len(i_chunks))
        i_chunk_end = min(i + chunk_size, F)
        PLAYER_POS_chunk = PLAYER_POS[i:i_chunk_end, ...]
        BALL_POS_chunk = BALL_POS[i:i_chunk_end, ...]
        phi_grid_chunk = phi_grid[i:i_chunk_end, ...]
        v0_grid_chunk = v0_grid[i:i_chunk_end, ...]
        passer_team_chunk = passer_team[i:i_chunk_end, ...]
        result = simulate_passes(PLAYER_POS_chunk, BALL_POS_chunk, phi_grid_chunk, v0_grid_chunk, passer_team_chunk, team_list)

        if full_result is None:
            full_result = result
        else:
            full_p_cum = np.concatenate([full_result.p_cum_att, result.p_cum_att], axis=0)
            full_p_density = np.concatenate([full_result.p_density_att, result.p_density_att], axis=0)
            full_phi = np.concatenate([full_result.phi_grid, result.phi_grid], axis=0)
            full_x0 = np.concatenate([full_result.x_grid, result.x_grid], axis=0)
            full_y0 = np.concatenate([full_result.y_grid, result.y_grid], axis=0)
            full_result = Result(
                p_cum_att=full_p_cum,
                p_density_att=full_p_density,
                phi_grid=full_phi,
                r_grid=full_result.r_grid,
                x_grid=full_x0,
                y_grid=full_y0
            )

    return full_result


def mask_out_of_play(simulation_result: Result) -> Result:
    x = simulation_result.x_grid
    y = simulation_result.y_grid

    on_pitch_mask = ((x >= -52.5) & (x <= 52.5) & (y >= -34) & (y <= 34))  # F x PHI x T

    simulation_result = simulation_result._replace(
        p_cum_att=np.where(on_pitch_mask, simulation_result.p_cum_att, 0),
        p_density_att=np.where(on_pitch_mask, simulation_result.p_density_att, 0)
    )
    return simulation_result


def add_xT_to_result(simulation_result: Result):
    import databallpy.models.utils
    import databallpy.events.base_event
    xt_model = databallpy.events.base_event.OPEN_PLAY_XT
    xt = databallpy.models.utils.get_xT_prediction(simulation_result.x_grid, simulation_result.y_grid, xt_model)

    simulation_result = simulation_result._replace(
        p_cum_att=simulation_result.p_cum_att * xt,
        p_density_att=simulation_result.p_density_att * xt,
    )

    return simulation_result
