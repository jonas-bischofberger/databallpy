import pandas as pd

# df_tracking = pd.DataFrame({
#     "frame_id": [0, 0, 0, 0, 0] + [1, 1, 1, 1, 1],
#     "player_id": ["A", "B", "C", "X", "Y", "ball"] * 2,
#     "team_id": [0, 0, 0, 1, 1, None] * 2,
#     "x": [-0.1, -10, -15, 15, 50, 0] + [-0.2, -9, -13, 16, 49],
#     "y": [0, 11, -14, 30, -1, 0] + [0.5, 10, -13, 30, -1],
#     "vx": [0.1, 1, 1.5, 1.5, -0.5, -5] + [0.2, 1.1, 1.3, 1.6, -0.9],
#     "vy": [0, 1, 0.6, -0.5, -1.1, 0] + [0.1, 1, 0.7, -0.5, -1.2],
# })
#
# df_passes = pd.DataFrame({
#     "frame_id": [0, 1],
#     "player_id": ["A", "B"],
#     "receiver_id": ["B", "X"],
#     "team_id": [0, 0],
#     "x": [-0.1, -10],
# })


import pandas as pd
import numpy as np

# Number of frames
n_frames = 20

# Generate smooth position data for each player over 20 frames
def generate_smooth_positions(start_x, start_y, vx, vy, n_frames):
    x_positions = [start_x + i * vx for i in range(n_frames)]
    y_positions = [start_y + i * vy for i in range(n_frames)]
    return x_positions, y_positions

# Player tracking data

# Player A (Team 0) smooth movement
x_A, y_A = generate_smooth_positions(start_x=-0.1, start_y=0, vx=0.1, vy=0.05, n_frames=n_frames)
# Player B (Team 0) smooth movement
x_B, y_B = generate_smooth_positions(start_x=-10, start_y=11, vx=0.2, vy=-0.1, n_frames=n_frames)
# Player C (Team 0) smooth movement
x_C, y_C = generate_smooth_positions(start_x=-15, start_y=-14, vx=0.3, vy=0.1, n_frames=n_frames)
# Player X (Team 1) smooth movement
x_X, y_X = generate_smooth_positions(start_x=15, start_y=30, vx=0.2, vy=0, n_frames=n_frames)
# Player Y (Team 1) smooth movement
x_Y, y_Y = generate_smooth_positions(start_x=50, start_y=-1, vx=-0.2, vy=0, n_frames=n_frames - 1)
# Ball smooth movement
x_ball, y_ball = generate_smooth_positions(start_x=0, start_y=0, vx=0.1, vy=0, n_frames=n_frames)


# Create tracking data for all players and ball
df_tracking = pd.DataFrame({
    "frame_id": list(range(n_frames)) * 4 + list(range(n_frames - 1)) + list(range(n_frames)),
    "player_id": ["A"] * n_frames + ["B"] * n_frames + ["C"] * n_frames + ["X"] * n_frames + ["Y"] * (n_frames - 1) + ["ball"] * n_frames,
    "team_id": [0] * n_frames + [0] * n_frames + [0] * n_frames + [1] * n_frames + [1] * (n_frames - 1) + [None] * n_frames,
    "x": x_A + x_B + x_C + x_X + x_Y + x_ball,
    "y": y_A + y_B + y_C + y_X + y_Y + y_ball,
    "vx": [0.1] * n_frames + [0.2] * n_frames + [0.3] * n_frames + [0.2] * n_frames + [-0.2] * (n_frames - 1) + [0.1] * n_frames,
    "vy": [0.05] * n_frames + [-0.1] * n_frames + [0.1] * n_frames + [0] * n_frames + [0] * (n_frames - 1) + [0] * n_frames,
})

# Passes data for 3 passes (2 successful, 1 failed)
df_passes = pd.DataFrame({
    "frame_id": [0, 6, 14],
    "player_id": ["A", "B", "C"],  # Players making the passes
    "receiver_id": ["B", "X", "Y"],  # Intended receivers
    "team_id": [0, 0, 0],  # Team of players making the passes
    "x": [-0.1, -9.6, -13.8],  # X coordinate where the pass is made
    "y": [0, 10.5, -12.9],  # Y coordinate where the pass is made
    "x_target": [-10, 15, 49],  # X target of the pass (location of receiver)
    "y_target": [11, 30, -1],  # Y target of the pass (location of receiver)
    "pass_outcome": ["successful", "failed", "failed"]  # Correct pass outcomes
})

print(df_tracking.head())
print(df_passes)
