"""Tests for the HED tag helpers."""

import unittest

import pandas as pd

from camstim_behavior_processing.nwb import hed_tags as H


class StimPresentationHedTest(unittest.TestCase):
    """Tests for :func:`stim_presentation_hed`."""

    def test_omitted(self):
        """Omitted flashes get the Unexpected tag regardless of name."""
        self.assertIn(
            "Unexpected", H.stim_presentation_hed("im000", True, True)
        )

    def test_image_change(self):
        """A natural image change uses Image and appends Target."""
        hed = H.stim_presentation_hed("im031", True, False)
        self.assertIn("(Image, Label/im031)", hed)
        self.assertTrue(hed.endswith("Target"))

    def test_gratings_no_change(self):
        """A gratings flash uses Grating and no Target when not a change."""
        hed = H.stim_presentation_hed("gratings_90", False, False)
        self.assertIn("(Grating, Label/gratings_90)", hed)
        self.assertNotIn("Target", hed)


class TrialHedTest(unittest.TestCase):
    """Tests for :func:`trial_hed`."""

    def _hed(self, **flags):
        """Return trial_hed for a row with the given outcome flags set."""
        return H.trial_hed(pd.Series(flags))

    def test_each_outcome(self):
        """Every outcome flag maps to its dedicated tag."""
        self.assertEqual(self._hed(hit=True), H.TRIAL_OUTCOME_HED["hit"])
        self.assertEqual(self._hed(miss=True), H.TRIAL_OUTCOME_HED["miss"])
        self.assertEqual(
            self._hed(false_alarm=True),
            H.TRIAL_OUTCOME_HED["false_alarm"],
        )
        self.assertEqual(
            self._hed(correct_reject=True),
            H.TRIAL_OUTCOME_HED["correct_reject"],
        )
        self.assertEqual(
            self._hed(aborted=True), H.TRIAL_OUTCOME_HED["aborted"]
        )
        self.assertEqual(
            self._hed(auto_rewarded=True),
            H.TRIAL_OUTCOME_HED["auto_rewarded"],
        )

    def test_no_outcome(self):
        """A row with no outcome flag falls back to no_outcome."""
        self.assertEqual(
            self._hed(hit=False), H.TRIAL_OUTCOME_HED["no_outcome"]
        )


class SweepStimHedTest(unittest.TestCase):
    """Tests for the passive SweepStim HED helpers."""

    def test_hed_safe_label(self):
        """Non-word characters are replaced with underscores."""
        self.assertEqual(H.hed_safe_label("a b.c-d"), "a_b_c_d")

    def test_movie_hed(self):
        """A movie clip HED nests a sanitised Movie label."""
        hed = H.sweepstim_movie_hed("clip one")
        self.assertIn("(Movie, Label/clip_one)", hed)

    def test_epoch_hed_spontaneous(self):
        """The spontaneous epoch maps to its fixed fragment."""
        self.assertEqual(
            H.sweepstim_epoch_hed("spontaneous"),
            H.SWEEPSTIM_SPONTANEOUS_EPOCH_HED,
        )

    def test_epoch_hed_passive(self):
        """The passive_viewing fallback maps to its fixed fragment."""
        self.assertEqual(
            H.sweepstim_epoch_hed("passive_viewing"),
            H.SWEEPSTIM_PASSIVE_EPOCH_HED,
        )

    def test_epoch_hed_clip(self):
        """A clip epoch nests both the task and Movie labels."""
        hed = H.sweepstim_epoch_hed("natural_movie_one")
        self.assertIn("Label/passive_viewing", hed)
        self.assertIn("(Movie, Label/natural_movie_one)", hed)


if __name__ == "__main__":
    unittest.main()
