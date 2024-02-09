import unittest

import pandas as pd

from databallpy.features import add_team_possession
from databallpy.utils.utils import MISSING_INT


class TestAddTeamPossession(unittest.TestCase):
    def setUp(self) -> None:
        self.tracking_data = pd.DataFrame(
            {
                "event_id": [MISSING_INT, 1, 6, MISSING_INT, 8, MISSING_INT],
                "ball_possession": [None, None, None, None, None, None],
            }
        )
        self.event_data = pd.DataFrame(
            {
                "event_id": [1, 3, 6, 7, 8],
                "databallpy_event": ["pass", "tackle", "pass", "interception", "pass"],
                "team_id": [1, 2, 1, 2, 2],
            }
        )

    def test_add_team_possession(self):
        add_team_possession(self.tracking_data, self.event_data, 1)
        self.assertEqual(
            self.tracking_data.ball_possession.tolist(),
            ["home", "home", "home", "home", "away", "away"],
        )

    def test_add_team_possession_no_event_id(self):
        with self.assertRaises(ValueError):
            add_team_possession(self.tracking_data, self.event_data, 22)

        self.tracking_data = self.tracking_data.copy().drop(columns=["event_id"])
        with self.assertRaises(ValueError):
            add_team_possession(self.tracking_data, self.event_data, 1)