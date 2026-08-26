"""Germany specialist: same FiLM ResUNet as Europe, smaller spatial domain.

The official Germany scoring box is 47-56N, 6-15E (37 x 37 cells). The model
sees a padded context window so weather can enter from the Atlantic/Alps:

    42.00-61.75N, 2.00W-21.75E  ->  80 x 96  (divisible by 8)

That window is a slice of the Europe crop already on disk, so no extra
dataset extract is needed. Train with official wind/temp scalars, loss
masked to the Germany box.

Serving: run this model on the 80 x 96 window, then feather its output
into the Europe (or global) field.
"""

from __future__ import annotations

from zeus_ml.models.europe_resunet import (  # noqa: F401
    CONTEXT_FEATURES,
    IN_CHANNELS,
    MAX_LEAD_HOURS,
    N_VARS,
    VARIABLE_WEIGHTS,
    EuropeOutput,
    EuropeResUNet,
    bracket_for_lead,
    build_context,
    cosine_solar_zenith,
    evaluate_climatology,
)

# Indices into the Europe crop (208 x 368, origin 28N / 40W).
GERMANY_LAT_SLICE = slice(56, 136)  # 80 cells, 42.00 .. 61.75 N
GERMANY_LON_SLICE = slice(152, 248)  # 96 cells, -2.00 .. 21.75 E
GERMANY_HEIGHT = 80
GERMANY_WIDTH = 96
# Official scoring box inside that window (not the context halo).
GERMANY_SCORE_LAT = (47.0, 56.0)
GERMANY_SCORE_LON = (6.0, 15.0)


class GermanyResUNet(EuropeResUNet):
    """Identical architecture; separate class so checkpoints stay named."""
