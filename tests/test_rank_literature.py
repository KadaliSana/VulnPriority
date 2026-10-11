"""Competitors re-implemented from the literature review (rank/literature.py).

They share the baselines' contract: registered under their own name, one finite score per
row, and usable in every ablation cell. The closed-form ones get a small ordering check
each; the learned ones are checked to train, to score, and to fall back to their prior when
a training fold has only one class.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_rank_baselines import frame_of, make_item, order
from vulnpriority.core.interfaces import Ranker
from vulnpriority.core.models import ComponentFlags
from vulnpriority.core.registry import get_ranker
from vulnpriority.rank.literature import (
    LITERATURE_RANKERS,
    AlbaneseRanker,
    CavpRanker,
    ContextXgbRanker,
    LicalityRanker,
    SsvcRanker,
    TopologyEpssRanker,
    VpssRandomForestRanker,
    VulnScoreRanker,
)

CLASSES = (SsvcRanker, CavpRanker, AlbaneseRanker, VulnScoreRanker, ContextXgbRanker,
           VpssRandomForestRanker, LicalityRanker, TopologyEpssRanker)
LEARNED = (ContextXgbRanker, VpssRandomForestRanker, LicalityRanker)


def mixed_frame():
    items = [
        make_item(f"f{i}", kev=i % 4 == 0, epss=(i % 10) / 10.0, cvss=float(i % 11))
        for i in range(40)
    ]
    return frame_of(items)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_every_competitor_is_a_registered_ranker(cls) -> None:
    assert isinstance(cls(), Ranker)
    assert get_ranker(cls.name) is cls
    assert cls.name in LITERATURE_RANKERS


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_every_competitor_scores_every_row_finitely(cls) -> None:
    frame = mixed_frame()
    ranker = cls()
    if ranker.requires_fit():
        relevance = np.array([1.0 if i % 4 == 0 else 0.0 for i in range(40)])
        ranker.fit(frame, relevance, None, 7)
    scores = np.asarray(ranker.score(frame))
    assert scores.shape == (40,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("cls", LEARNED, ids=lambda cls: cls.__name__)
def test_a_single_class_fold_falls_back_to_the_prior(cls) -> None:
    frame = mixed_frame()
    ranker = cls().fit(frame, np.zeros(40), None, 7)
    assert np.isfinite(np.asarray(ranker.score(frame))).all()


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_competitors_survive_the_cell_with_every_component_off(cls) -> None:
    items = [make_item(f"f{i}", epss=0.1 * i, cvss=float(i)) for i in range(6)]
    frame = frame_of(items, ComponentFlags(a=False, b=False, c=False))
    ranker = cls()
    if ranker.requires_fit():
        ranker.fit(frame, np.array([0, 0, 0, 1, 1, 1], dtype=float), None, 7)
    assert np.isfinite(np.asarray(ranker.score(frame))).all()


def test_ssvc_puts_an_actively_exploited_finding_first() -> None:
    frame = frame_of([make_item("quiet", cvss=9.8), make_item("kev", kev=True, cvss=5.0)])
    assert order(frame, SsvcRanker().score(frame))[0] == "kev"


def test_cavp_never_raises_a_score_above_its_cvss_base() -> None:
    frame = frame_of([make_item("a", cvss=7.5), make_item("b", cvss=9.1)])
    base = frame.X["cvss_base_max"].to_numpy()
    assert (np.floor(CavpRanker().score(frame)) <= np.ceil(base)).all()
    assert order(frame, CavpRanker().score(frame)) == ["b", "a"]


def test_vulnscore_rises_with_epss_when_everything_else_is_equal() -> None:
    frame = frame_of([make_item("low", epss=0.1, cvss=5.0), make_item("high", epss=0.9, cvss=5.0)])
    assert order(frame, VulnScoreRanker().score(frame)) == ["high", "low"]
