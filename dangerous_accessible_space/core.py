import numpy as np
import pandas as pd
import streamlit as st
import math
import scipy.integrate
import collections

import dangerous_accessible_space.motion

_result_fields = [
    "poss_cum_att",  # F x PHI x T
    "prob_cum_att",  # F x PHI x T
    "poss_density_att",  # F x PHI x T
    "prob_density_att",  # F x PHI x T
    "poss_cum_def",  # F x PHI x T
    "prob_cum_def",  # F x PHI x T
    "poss_density_def",  # F x PHI x T
    "prob_density_def",  # F x PHI x T
    "phi_grid",  # PHI
    "r_grid",  # T
    "x_grid",  # F x PHI x T
    "y_grid",  # F x PHI x T
]
Result = collections.namedtuple("Result", _result_fields, defaults=[None] * len(_result_fields))

# DEFAULT_B0 = -5
# DEFAULT_B1 = -20
# DEFAULT_PASS_START_LOCATION_OFFSET = 0
# DEFAULT_TIME_OFFSET_BALL = 0
# DEFAULT_SECONDS_TO_INTERCEPT = 0.1
# DEFAULT_TOL_DISTANCE = 5
# DEFAULT_PLAYER_VELOCITY = 7
# DEFAULT_KEEP_INERTIAL_VELOCITY = True
# DEFAULT_A_MAX = 14.256003027575932
# DEFAULT_V_MAX = 12.865546440947865
# DEFAULT_USE_MAX = False
# DEFAULT_USE_APPROX_TWO_POINT = True
# DEFAULT_INERTIAL_SECONDS = 0.2
# DEFAULT_RADIAL_GRIDSIZE = 3




#             ### best (b0=-5, b1=-20, 9/0.2/5 MM, start=3, num=80, stop=22.5)
#             fields = LowSpearmanLike(
# #            b0=-5,
# #             b1=-1,
# #                 b0=-1,
#             b0_att=0,
#             b0_def=0,  # -0.5 best so far
#
#
#             b1_att=-25,
#             b1_def=-25,
#             mm_player=models.motion.ApproxTwoPoint(
#                 use_max=False, velocity=7,
#                 inertial_seconds=0.25,
#                 keep_inertial_velocity=True,
#                 tol_distance=None,
#             ),
#             # mm_player=models.motion.ConstVel(7),
#             mm_ball=models.motion.NoForce(),
#             pass_selector=models.pass_selection.low.Uniform(
#                 n_angles=4 * 30,  # Anzahl der Winkel durch 4 teilbar => ca. isotrop?
#                 # velocity_range=np.linspace(start=3, stop=22.5, num=7, endpoint=True),
#                 velocity_range=np.linspace(start=3, stop=22.5, num=60, endpoint=True),
#                 mode="poss"  # TODO hier prob/poss unterscheidung rein
#             ),
#             ).get_field_matrix(
#                 game, unit="dp", offside=False, single_half=0, single_fr=fr, gridsize=2,
#             )

DEFAULT_B0 = -1.3075312012275244
DEFAULT_B1 = -65.57184250749606
DEFAULT_PASS_START_LOCATION_OFFSET = 0#0.2821895970952328
DEFAULT_TIME_OFFSET_BALL = 0#-0.09680365586691105
DEFAULT_SECONDS_TO_INTERCEPT = 1.1650841463114299
DEFAULT_TOL_DISTANCE = 2.5714050933456036
DEFAULT_PLAYER_VELOCITY = 3.984451038279267
DEFAULT_KEEP_INERTIAL_VELOCITY = True
DEFAULT_A_MAX = 14.256003027575932
DEFAULT_V_MAX = 12.865546440947865
DEFAULT_USE_MAX = True
DEFAULT_USE_APPROX_TWO_POINT = False #True
DEFAULT_INERTIAL_SECONDS = 0.6164609802178712
DEFAULT_RADIAL_GRIDSIZE = 3
#

def sigmoid(x):
    return 0.5 * (x / (1 + np.abs(x)) + 1)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def integrate_trapezoid(y, x):
    return scipy.integrate.cumulative_trapezoid(y=y, x=x, initial=0, axis=-1)  # * radial_gridsize # F x V0 x PHI x T


def simulate_passes(
    PLAYER_POS,  # F x P x 4[x, y, vx, vy], player positions
    BALL_POS,  # F x 2[x, y], ball positions
    phi_grid,  # F x PHI, pass angles
    v0_grid,  # F x V0, pass speeds
    passer_teams,  # F, team of passers
    player_teams,  # P, player teams
    players=None,  # P, players
    passers_to_exclude=None,  # F, passers IF we want to exclude them

    pass_start_location_offset=DEFAULT_PASS_START_LOCATION_OFFSET,
    time_offset_ball=DEFAULT_TIME_OFFSET_BALL,
    radial_gridsize=DEFAULT_RADIAL_GRIDSIZE,
    seconds_to_intercept=DEFAULT_SECONDS_TO_INTERCEPT,
    b0=DEFAULT_B0,
    b1=DEFAULT_B1,
    player_velocity=DEFAULT_PLAYER_VELOCITY,
    keep_inertial_velocity=DEFAULT_KEEP_INERTIAL_VELOCITY,
    use_max=DEFAULT_USE_MAX,
    v_max=DEFAULT_V_MAX,
    a_max=DEFAULT_A_MAX,
    inertial_seconds=DEFAULT_INERTIAL_SECONDS,
    tol_distance=DEFAULT_TOL_DISTANCE,
    use_approx_two_point=DEFAULT_USE_APPROX_TWO_POINT,
) -> Result:
    """ Calculate the pass simulation model - Core functionality of this package """
    _assert_matrices_validity(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams)

    ### 1. Calculate ball trajectory
    # 1.1 Calculate spatial grid
    D_BALL_SIM = np.linspace(start=pass_start_location_offset, stop=150, num=math.ceil(150 / radial_gridsize))  # T

    # 1.2 Calculate temporal grid
    T_BALL_SIM = dangerous_accessible_space.motion.constant_velocity_time_to_arrive_1d(
        x=D_BALL_SIM[0], v=v0_grid[:, :, np.newaxis], x_target=D_BALL_SIM[np.newaxis, np.newaxis, :],
    )  # F x V0 x T
    T_BALL_SIM += time_offset_ball

    # 1.3 Calculate 2D points along ball trajectory
    cos_phi, sin_phi = np.cos(phi_grid), np.sin(phi_grid)  # F x PHI
    X_BALL_SIM = BALL_POS[:, 0][:, np.newaxis, np.newaxis] + cos_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T
    Y_BALL_SIM = BALL_POS[:, 1][:, np.newaxis, np.newaxis] + sin_phi[:, :, np.newaxis] * D_BALL_SIM[np.newaxis, np.newaxis, :]  # F x PHI x T

    ### 2 Calculate player interception rates
    # 2.1 Calculate time to arrive for each player along ball trajectory
    if use_approx_two_point:
        TTA_PLAYERS = dangerous_accessible_space.motion.approx_two_point_time_to_arrive(  # F x P x PHI x T
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
            y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            vx=PLAYER_POS[:, :, 2][:, :, np.newaxis, np.newaxis],
            vy=PLAYER_POS[:, :, 3][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :],
            y_target=Y_BALL_SIM[:, np.newaxis, :, :],

            # Parameters
            use_max=use_max, velocity=player_velocity, keep_inertial_velocity=keep_inertial_velocity, v_max=v_max,
            a_max=a_max, inertial_seconds=inertial_seconds, tol_distance=tol_distance,
        )
    else:
        TTA_PLAYERS = dangerous_accessible_space.motion.constent_velocity_time_to_arrive(  # F x P x PHI x T
            x=PLAYER_POS[:, :, 0][:, :, np.newaxis, np.newaxis],
            y=PLAYER_POS[:, :, 1][:, :, np.newaxis, np.newaxis],
            x_target=X_BALL_SIM[:, np.newaxis, :, :],
            y_target=Y_BALL_SIM[:, np.newaxis, :, :],
            player_velocity=player_velocity,
        )

    if passers_to_exclude is not None:
        i_passers_to_exclude = np.array([list(players).index(passer) for passer in passers_to_exclude])
        i_frames = np.arange(TTA_PLAYERS.shape[0])
        TTA_PLAYERS[i_frames, i_passers_to_exclude, :, :] = np.inf  # F x P x PHI x T

    TTA_PLAYERS = np.nan_to_num(TTA_PLAYERS, nan=np.inf)  # Handle players not participating in the game by setting their TTA to infinity
    # st.write("T_BALL_SIM[fr=0, PHI=0, T=:]", T_BALL_SIM.shape)
    # st.write(T_BALL_SIM[0, 0, :])
    # st.write("TTA_PLAYERS[fr=0, P=:, PHI=0, T=:]", TTA_PLAYERS.shape)
    # st.write(TTA_PLAYERS[0, :, 0, :])

    # 2.2 Transform time to arrive into interception rates
    seconds_to_intercept = 15
    X = TTA_PLAYERS[:, :, np.newaxis, :, :] - T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x P x PHI x T - F x PHI x T = F x P x V0 x PHI x T
    # TIME_TO_CONTROL = np.maximum(0, (1 - np.exp(seconds_to_intercept * X)))  # F x P x V0 x PHI x T
    # st.write("X[fr=0, P=:, V0=0, PHI=0, T=:]", X.shape)
    # st.write(X[0, :, 0, 0, :])
    X[:] = b0 + b1 * X  # 1 + 1 * F x P x V0 x PHI x T = F x P x V0 x PHI x T
    X[:] = sigmoid(X)
    X = np.nan_to_num(X, nan=0)  # F x P x V0 x PHI x T
    # TIME_TO_CONTROL = np.nan_to_num(TIME_TO_CONTROL, nan=0)  # F x P x V0 x PHI x T
    # st.write("TIME_TO_CONTROL[fr=0, P=:, V0=0, PHI=0, T=:]", TIME_TO_CONTROL.shape)
    # st.write(TIME_TO_CONTROL[0, :, 0, 0, :])
    X = X #* TIME_TO_CONTROL

    # st.write("X(sigmoid)[fr=0, P=:, V0=0, PHI=0, T=:]", X.shape)
    # st.write(X[0, :, 0, 0, :])
    # st.write("seconds_to_intercept", seconds_to_intercept)
    # ar_time = X / seconds_to_intercept  # F x P x V0 x PHI x T, interception rate / DR[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    # ar_time = X / seconds_to_intercept  # F x P x V0 x PHI x T, interception rate / DR[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    # st.write("ar_time[fr=0, P=:, V0=0, PHI=0, T=:]", ar_time.shape, ar_time.min(), ar_time.mean(), ar_time.max())
    # st.write(ar_time[0, :, 0, 0, :])
    DT = T_BALL_SIM[:, :, 1] - T_BALL_SIM[:, :, 0]  # F x V0
    ar_time = X / DT[:, np.newaxis, :, np.newaxis, np.newaxis]  # F x P x V0 x PHI x T
    # st.write("ar_time[fr=0, P=:, V0=0, PHI=0, T=:]", ar_time.shape, ar_time.min(), ar_time.mean(), ar_time.max())
    # st.write(ar_time[0, :, 0, 0, :])

    # st.write("DT", DT.shape, DT)
    # ar_space = ar_time * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / radial_gridsize  # F x P x V0 x PHI x T
    # st.write("ar_space[fr=0, P=:, V0=0, PHI=0, T=:]", ar_space.shape, ar_space.min(), ar_space.mean(), ar_space.max())
    # st.write(ar_space[0, :, 0, 0, :])
    # ar_time = ar_space

    ## 3. Use interception rates to calculate probabilities
    # 3.1 Sums of interception rates over players
    sum_ar = np.nansum(ar_time, axis=1)  # F x V0 x PHI x T
    # st.write("sum_ar[fr=0, v0=0, phi=0, T=:]", sum_ar.shape)
    # st.write(sum_ar[0, 0, 0, :])

    # poss-specific
    player_is_attacking = player_teams[np.newaxis, :] == passer_teams[:, np.newaxis]  # F x P
    x = np.where(player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    y = np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis], ar_time, 0)  # F x P x V0 x PHI x T
    sum_ar_att = np.nansum(x, axis=1)  # F x V0 x PHI x T
    sum_ar_def = np.nansum(y, axis=1)  # F x V0 x PHI x T

    # poss-specific
    int_sum_ar = integrate_trapezoid(y=sum_ar, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_att = integrate_trapezoid(y=sum_ar_att, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    int_sum_ar_def = integrate_trapezoid(y=sum_ar_def, x=T_BALL_SIM[:, :, np.newaxis, :])  # F x V0 x PHI x T
    # st.write("int_sum_ar[fr=0, v0=0, phi=0, T=:]", int_sum_ar.shape)
    # st.write(int_sum_ar[0, 0, 0, :])

    # Cumulative probabilities from integrals
    p0_cum = np.exp(-int_sum_ar) #if "prob" in ptypes else None  # F x V0 x PHI x T, cumulative probability that no one intercepted
    p0_cum_only_att = np.exp(-int_sum_ar_att) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_cum_only_def = np.exp(-int_sum_ar_def) #if "poss" in ptypes else None  # F x V0 x PHI x T
    p0_only_opp = np.where(
        player_is_attacking[:, :, np.newaxis, np.newaxis, np.newaxis],
        p0_cum_only_def[:, np.newaxis, :, :, :], p0_cum_only_att[:, np.newaxis, :, :, :]
    ) #if "poss" in ptypes else None  # F x P x V0 x PHI x T
    # st.write("p0_cum[fr=0, v0=0, phi=0, T=:]", p0_cum.shape)
    # st.write(p0_cum[0, 0, 0, :])
    # st.write("p0_cum_only_att[fr=0, v0=0, phi=0, T=:]", p0_cum_only_att.shape)
    # st.write(p0_cum_only_att[0, 0, 0, :])
    # st.write("p0_cum_only_def[fr=0, v0=0, phi=0, T=:]", p0_cum_only_def.shape)
    # st.write(p0_cum_only_def[0, 0, 0, :])
    # st.write("p0_only_opp[fr=0, p=:, v0=0, phi=0, T=:]", p0_only_opp.shape)
    # st.write(p0_only_opp[0, :, 0, 0, :])

    # Individual probability densities
    dpr_over_dt = p0_cum[:, np.newaxis, :, :, :] * ar_time  # if "prob" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_prob = integrate_trapezoid(  # F x P x V0 x PHI x T, cumulative probability that player P intercepted
        y=dpr_over_dt,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    dpr_poss_over_dt = p0_only_opp * ar_time  # if "poss" in ptypes else None  # F x P x V0 x PHI x T
    pr_cum_poss = integrate_trapezoid(  # F x P x V0 x PHI x T
        y=dpr_poss_over_dt,  # F x P x V0 x PHI x T
        x=T_BALL_SIM[:, np.newaxis, :, np.newaxis, :]  # F x V0 x T
    )  # if add_receiver else None

    dp0_over_dt = -p0_cum * sum_ar  # F x V0 x PHI x T

    # Go from dt -> dx
    DX = D_BALL_SIM[1] - D_BALL_SIM[0]
    dpr_over_dx = dpr_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX  # F x P x V0 x PHI x T
    dpr_poss_over_dx = dpr_poss_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX  # F x P x V0 x PHI x T
    dp0_over_dx = dp0_over_dt * DT[:, np.newaxis, :, np.newaxis, np.newaxis] / DX

    # Aggregate over v0
    # dpr_over_dx_vagg_prob = np.average(dpr_over_dt, axis=2)  # if "prob" in ptypes else None  # F x P x PHI x T, Take the average over all V0 in v0_grid
    dpr_over_dx_vagg_prob = np.average(dpr_over_dx, axis=2)  # if "prob" in ptypes else None  # F x P x PHI x T, Take the average over all V0 in v0_grid
    # dpr_over_dx_vagg_poss = np.max(dpr_poss_over_dt, axis=2)  # if "poss" in ptypes else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269
    dpr_over_dx_vagg_poss = np.max(dpr_poss_over_dx, axis=2)  # if "poss" in ptypes else None  # F x P x PHI x T, np.max not supported yet with numba using axis https://github.com/numba/numba/issues/1269
    dp0_over_dx_vagg = np.average(dp0_over_dx, axis=1)  # F x PHI x T

    # Normalize 3/4: poss density
    dpr_over_dx_vagg_poss_times_dx = dpr_over_dx_vagg_poss * DX  # F x P x PHI x T
    num_max = np.max(dpr_over_dx_vagg_poss_times_dx, axis=(1, 3))  # F
    dpr_over_dx_vagg_poss = dpr_over_dx_vagg_poss / num_max[:, np.newaxis, :, np.newaxis]
    # st.write("dpr_over_dx_vagg_poss", dpr_over_dx_vagg_poss.shape, np.min(dpr_over_dx_vagg_poss), np.max(dpr_over_dx_vagg_poss))

    # Normalize 2/4: prob density
    dpr_over_dx_vagg_prob_sum = np.sum(dpr_over_dx_vagg_prob * radial_gridsize, axis=(1, 3))  # F x PHI
    dpr_over_dx_vagg_prob = dpr_over_dx_vagg_prob / dpr_over_dx_vagg_prob_sum[:, np.newaxis, :, np.newaxis]  # F x P x PHI x T

    p0_cum_vagg = np.mean(p0_cum, axis=1)  # if add_receiver else None  # F x PHI x T
    pr_cum_prob_vagg = np.mean(pr_cum_prob, axis=2)  # if add_receiver else None  # F x P x PHI x T
    pr_cum_poss_vagg = np.max(pr_cum_poss, axis=2)  # if add_receiver else None  # F x P x V0 x PHI x T -> F x P x V0 x PHI x T
    # st.write("pr_cum_prob_vagg", pr_cum_prob_vagg.shape, np.max(pr_cum_prob_vagg))
    # st.write(pr_cum_prob_vagg[0, :, 0, :])

    # pr_cum_prob_vagg = np.minimum(pr_cum_prob_vagg, 1)
    # pr_cum_poss_vagg = np.minimum(pr_cum_poss_vagg, 1)
    # dpr_over_dx_vagg_prob = np.minimum(dpr_over_dx_vagg_prob, 1/radial_gridsize)
    # dpr_over_dx_vagg_poss = np.minimum(dpr_over_dx_vagg_poss, 1/radial_gridsize)

    dpr_over_dx_vagg_att_prob = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_prob, 0), axis=1)  # F x PHI x T
    dpr_over_dx_vagg_def_prob = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_prob, 0), axis=1)  # F x PHI x T
    dpr_over_dx_vagg_att_poss = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_poss, 0), axis=1)  #if add_receiver else None  # F x PHI x T
    dpr_over_dx_vagg_def_poss = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], dpr_over_dx_vagg_poss, 0), axis=1)  #if add_receiver else None  # F x PHI x T

    pr_cum_att = np.nansum(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_def = np.nansum(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_prob_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_att = np.nanmax(np.where(player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T
    pr_cum_poss_def = np.nanmax(np.where(~player_is_attacking[:, :, np.newaxis, np.newaxis], pr_cum_poss_vagg, 0), axis=1) #if add_receiver else None  # F x PHI x T

    # pr_cum_poss_att = np.minimum(pr_cum_poss_att, 1)
    # pr_cum_poss_def = np.minimum(pr_cum_poss_def, 1)

    # dpr_over_dx_vagg_att_poss[..., 0] = 1 / radial_gridsize
    # dpr_over_dx_vagg_def_poss[..., 0] = 1 / radial_gridsize
    # dpr_over_dx_vagg_def_poss[..., 0] = 1 / radial_gridsize
    # dpr_over_dx_vagg_def_prob[..., 0] = 1 / radial_gridsize

    # normalize 1/4: prob cum
    p_sum = pr_cum_att + pr_cum_def + p0_cum_vagg
    pr_cum_att = pr_cum_att / p_sum
    pr_cum_def = pr_cum_def / p_sum

    # normalize 3/4: poss cum

    # st.write("dpr_over_dx_vagg_att_poss", dpr_over_dx_vagg_att_poss.shape, np.min(dpr_over_dx_vagg_att_poss), np.max(dpr_over_dx_vagg_att_poss))
    # st.write("dpr_over_dx_vagg_def_poss", dpr_over_dx_vagg_def_poss.shape, np.min(dpr_over_dx_vagg_def_poss), np.max(dpr_over_dx_vagg_def_poss))
    # st.write("dpr_over_dx_vagg_att_prob", dpr_over_dx_vagg_att_prob.shape, np.min(dpr_over_dx_vagg_att_prob), np.max(dpr_over_dx_vagg_att_prob))
    # st.write("dpr_over_dx_vagg_def_prob", dpr_over_dx_vagg_def_prob.shape, np.min(dpr_over_dx_vagg_def_prob), np.max(dpr_over_dx_vagg_def_prob))

    pr_cum_poss_att_max = np.maximum.accumulate(dpr_over_dx_vagg_att_poss, axis=2) * radial_gridsize  # possibility CDF uses cummax instead of cumsum to emerge from PDF
    pr_cum_poss_def_max = np.maximum.accumulate(dpr_over_dx_vagg_def_poss, axis=2) * radial_gridsize

    # st.write("poss_cum_att", np.min(pr_cum_poss_att_max), np.max(pr_cum_poss_att_max))
    # st.write("prob_cum_att", np.min(pr_cum_att), np.max(pr_cum_att))
    # st.write("poss_density_att", np.min(dpr_over_dx_vagg_att_poss), np.max(dpr_over_dx_vagg_att_poss))
    # st.write("prob_density_att", np.min(dpr_over_dx_vagg_att_prob), np.max(dpr_over_dx_vagg_att_prob))
    # st.write("poss_cum_def", np.min(pr_cum_poss_def_max), np.max(pr_cum_poss_def_max))
    # st.write("prob_cum_def", np.min(pr_cum_def), np.max(pr_cum_def))
    # st.write("poss_density_def", np.min(dpr_over_dx_vagg_def_poss), np.max(dpr_over_dx_vagg_def_poss))
    # st.write("prob_density_def", np.min(dpr_over_dx_vagg_def_prob), np.max(dpr_over_dx_vagg_def_prob))

    result = Result(
        poss_cum_att=pr_cum_poss_att_max,  # F x PHI x T
        prob_cum_att=pr_cum_att,  # F x PHI x T
        poss_density_att=dpr_over_dx_vagg_att_poss,  # F x PHI x T
        prob_density_att=dpr_over_dx_vagg_att_prob,  # F x PHI x T
        poss_cum_def=pr_cum_poss_def_max,  #pr_cum_poss_def,  # F x PHI x T
        prob_cum_def=pr_cum_def,  # F x PHI x T
        poss_density_def=dpr_over_dx_vagg_def_poss,  # F x PHI x T
        prob_density_def=dpr_over_dx_vagg_def_prob,  # F x PHI x T

        phi_grid=phi_grid, r_grid=D_BALL_SIM, x_grid=X_BALL_SIM, y_grid=Y_BALL_SIM
    )

    return result


def _assert_matrices_validity(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_team, team_list):
    F = PLAYER_POS.shape[0]
    assert F == BALL_POS.shape[0]
    assert F == phi_grid.shape[0]
    assert F == v0_grid.shape[0]
    assert F == passer_team.shape[0], f"Dimension F is {F} (from PLAYER_POS: {PLAYER_POS.shape}), but passer_team shape is {passer_team.shape}"
    P = PLAYER_POS.shape[1]
    assert P == team_list.shape[0]
    assert PLAYER_POS.shape[2] >= 4  # >= or = ?
    assert BALL_POS.shape[1] >= 2  # ...


def simulate_passes_chunked(
    PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams, players=None, passers_to_exclude=None,
    chunk_size=30
) -> Result:
    _assert_matrices_validity(PLAYER_POS, BALL_POS, phi_grid, v0_grid, passer_teams, player_teams)

    F = PLAYER_POS.shape[0]

    i_chunks = list(np.arange(0, F, chunk_size))
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
        passer_team_chunk = passer_teams[i:i_chunk_end, ...]

        result = simulate_passes(
            PLAYER_POS_chunk, BALL_POS_chunk, phi_grid_chunk, v0_grid_chunk, passer_team_chunk, player_teams, players,
            passers_to_exclude
        )

        if full_result is None:
            full_result = result
        else:
            full_p_cum = np.concatenate([full_result.prob_cum_att, result.prob_cum_att], axis=0)
            full_poss_cum = np.concatenate([full_result.poss_cum_att, result.poss_cum_att], axis=0)
            full_p_density = np.concatenate([full_result.poss_density_att, result.poss_density_att], axis=0)
            full_prob_density = np.concatenate([full_result.prob_density_att, result.prob_density_att], axis=0)
            full_p_cum_def = np.concatenate([full_result.prob_cum_def, result.prob_cum_def], axis=0)
            full_poss_cum_def = np.concatenate([full_result.poss_cum_def, result.poss_cum_def], axis=0)
            full_p_density_def = np.concatenate([full_result.poss_density_def, result.poss_density_def], axis=0)
            full_prob_density_def = np.concatenate([full_result.prob_density_def, result.prob_density_def], axis=0)
            full_phi = np.concatenate([full_result.phi_grid, result.phi_grid], axis=0)
            full_x0 = np.concatenate([full_result.x_grid, result.x_grid], axis=0)
            full_y0 = np.concatenate([full_result.y_grid, result.y_grid], axis=0)
            full_result = Result(
                poss_cum_att=full_poss_cum,
                prob_cum_att=full_p_cum,
                poss_density_att=full_p_density,
                prob_density_att=full_prob_density,
                poss_cum_def=full_poss_cum_def,
                prob_cum_def=full_p_cum_def,
                poss_density_def=full_p_density_def,
                prob_density_def=full_prob_density_def,
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
        prob_cum_att=np.where(on_pitch_mask, simulation_result.prob_cum_att, 0),
        poss_cum_att=np.where(on_pitch_mask, simulation_result.poss_cum_att, 0),
        poss_density_att=np.where(on_pitch_mask, simulation_result.poss_density_att, 0),
        prob_density_att=np.where(on_pitch_mask, simulation_result.prob_density_att, 0),
        poss_cum_def=np.where(on_pitch_mask, simulation_result.prob_cum_def, 0),
        prob_cum_def=np.where(on_pitch_mask, simulation_result.prob_cum_def, 0),
        poss_density_def=np.where(on_pitch_mask, simulation_result.poss_density_def, 0),
        prob_density_def=np.where(on_pitch_mask, simulation_result.prob_density_def, 0),
    )
    return simulation_result


def aggregate_surface_area(result):
    result = mask_out_of_play(result)

    # Get r-part of area elements
    r_grid = result.r_grid  # T

    r_lower_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_lower_bounds[1:] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and previous element
    r_lower_bounds[0] = r_grid[0]  # Set lower bound for the first element

    r_upper_bounds = np.zeros_like(r_grid)  # Initialize with zeros
    r_upper_bounds[:-1] = (r_grid[:-1] + r_grid[1:]) / 2  # Midpoint between current and next element
    r_upper_bounds[-1] = r_grid[-1]  # Arbitrarily high upper bound for the last element

    dr = r_upper_bounds - r_lower_bounds  # T

    # Get phi-part of area elements
    phi_grid = result.phi_grid  # F x PHI

    # dA = np.diff(phi_grid, axis=1) * r_grid[:, np.newaxis]  # F x PHI-1 x T

    phi_lower_bounds = np.zeros_like(phi_grid)  # F x PHI
    phi_lower_bounds[:, 1:] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and previous element
    phi_lower_bounds[:, 0] = phi_grid[:, 0]

    phi_upper_bounds = np.zeros_like(phi_grid)  # Initialize with zeros
    phi_upper_bounds[:, :-1] = (phi_grid[:, :-1] + phi_grid[:, 1:]) / 2  # Midpoint between current and next element
    phi_upper_bounds[:, -1] = phi_grid[:, -1]  # Arbitrarily high upper bound for the last element

    dphi = phi_upper_bounds - phi_lower_bounds  # F x PHI

    outer_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_upper_bounds[np.newaxis, np.newaxis, :]**2)  # T
    inner_bound_circle_slice_area = dphi[:, :, np.newaxis]/(2*np.pi) * (np.pi * r_lower_bounds[np.newaxis, np.newaxis, :]**2)  # T

    dA = outer_bound_circle_slice_area - inner_bound_circle_slice_area  # F x PHI x T

    p = result.poss_density_att * dr

    AS = np.sum(p * dA, axis=(1, 2))  # F

    return AS

    # xt = get_xT_prediction(x, y, THROW_IN_XT)


def add_xT_to_result(simulation_result: Result):
    import databallpy.models.utils
    import databallpy.events.base_event
    xt_model = databallpy.events.base_event.OPEN_PLAY_XT
    xt = databallpy.models.utils.get_xT_prediction(simulation_result.x_grid, simulation_result.y_grid, xt_model)

    simulation_result = simulation_result._replace(
        poss_cum_att=simulation_result.poss_cum_att * xt,
        prob_cum_att=simulation_result.prob_cum_att * xt,
        poss_density_att=simulation_result.poss_density_att * xt,
        prob_density_att=simulation_result.prob_density_att * xt,
        poss_cum_def=simulation_result.prob_cum_def * xt,
        prob_cum_def=simulation_result.prob_cum_def * xt,
        poss_density_def=simulation_result.poss_density_def * xt,
        prob_density_def=simulation_result.prob_density_def * xt,
    )

    return simulation_result
