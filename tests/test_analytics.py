import sys
from pathlib import Path
import unittest


sys.path.insert(
    0,
    str(
        Path(__file__)
        .parent
        .parent
        / "src"
    )
)


from analytics import (
    get_side,
    crossing_event
)


class TestAnalytics(unittest.TestCase):

    def test_above_line(self):

        self.assertEqual(
            get_side(
                100,
                200,
                15
            ),
            "above"
        )


    def test_below_line(self):

        self.assertEqual(
            get_side(
                300,
                200,
                15
            ),
            "below"
        )


    def test_dead_zone(self):

        self.assertIsNone(
            get_side(
                205,
                200,
                15
            )
        )


    def test_entered(self):

        self.assertEqual(
            crossing_event(
                "above",
                "below",
                False,
                False
            ),
            "entered"
        )


    def test_exited(self):

        self.assertEqual(
            crossing_event(
                "below",
                "above",
                False,
                False
            ),
            "exited"
        )


    def test_no_duplicate_entry(self):

        self.assertIsNone(
            crossing_event(
                "above",
                "below",
                True,
                False
            )
        )


if __name__ == "__main__":
    unittest.main()
    