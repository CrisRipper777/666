"""Isolated V1 MAG interaction study; not part of the benchmark model registry."""

from .splits import StudySplit, make_study_split, mask_test_labels

__all__ = ["StudySplit", "make_study_split", "mask_test_labels"]
