"""The vocabulary the spectrum figures share -- the source's lines, the colours the
curves are drawn in, and the rules that set a panel's y range.

Every one of these was defined in whichever figure script needed it first and imported
from there by the others, which made a figure script a library as well. They are
constants and small rules about display, so they belong in neither.
"""
import numpy as np
from scipy.signal import medfilt

# Haro 11's systemic redshift, measured from the data rather than assumed: the
# flux-weighted centroid of Hb, [O III] 4959 and 5007, Ha and [S II] 6716 over the
# brightest 1% of p05's galaxy spaxels, in ESO's cube so that our own subtraction
# cannot be what sets it. The five agree to +-0.00003, which is a tenth of a channel.
#
# It marks lines on figures and picks the channels --exclude-source-lines drops; the
# pipeline itself never sees it, taking its mask windows as observed wavelengths. One
# definition because a copy per script is a number that can drift between two figures
# of the same galaxy.
Z_HARO = 0.02064
# Rest wavelengths in air. Both halves of the [O III] doublet are marked because they
# share an upper level, so the transition probabilities alone fix the ratio at
# 5007/4959 = 2.98 -- a zone where it is not about 3 has a problem in the subtraction
# or the fit, not in the physics.
LINES = [("Hb", 4861.3), ("[O III] 4959", 4958.9), ("[O III] 5007", 5006.8),
         ("Ha", 6562.8), ("[S II]", 6716.4)]
# Pale red for the line markers -- a colour neither the grid nor the zero line uses.
# Grey would match the grid, whose 5000 A line falls beside the redshifted Hb marker.
C_LINE = "#f4a3a3"
# Transparency for the full-height marker: it sits under the peak it names, and at full
# strength the two are one stroke. A paler colour would lose the hue.
A_LINE = 0.45
# Ours and the reference, and the zero line they are read against. The same pair in
# every comparison figure, so a colour means the same thing from one to the next.
C_OURS, C_ESO, C_ZERO = "#1f77b4", "#e8710a", "0.55"


def despiked_range(y):
    """The range a curve occupies, with single-channel excursions left out.

    A dead or hot channel sits orders of magnitude from its neighbours, and a 3-channel
    median removes it while keeping a spectrally resolved emission line, which is
    several channels wide. The range is then extended back to any raw value within one
    full span of it, so a real line tip is not cut by a rule aimed at single channels.
    """
    m = medfilt(y, 3)
    lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
    span = max(hi - lo, 1e-9)
    near = y[(y >= lo - span) & (y <= hi + span)]
    if near.size:
        lo, hi = min(lo, float(near.min())), max(hi, float(near.max()))
    return lo, hi


def panel_ylim(spec, pad=0.06):
    """`despiked_range` with room left above and below.

    The same rule, plus a margin: a line tip drawn exactly on the frame reads as
    clipped even when the whole of it is inside.
    """
    lo, hi = despiked_range(spec)
    m = pad * max(hi - lo, 1e-9)
    return lo - m, hi + m


def robust_range(y, pct=0.5, pad=0.35):
    """A y range set by the spectrum, not by its worst channel.

    Percentiles rather than min/max: a few dead or hot channels would stretch the axis
    until everything real is flat on zero. Zero stays inside, being what these spectra
    are read against.
    """
    lo, hi = np.percentile(y[np.isfinite(y)], [pct, 100 - pct])
    m = pad * max(hi - lo, 1e-9)
    return min(lo - m, 0.0), max(hi + m, 0.0)
