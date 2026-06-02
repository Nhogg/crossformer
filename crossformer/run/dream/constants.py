from __future__ import annotations

import numpy as np

KP_CONF_THRESHOLD = 0.03
KP_SMOOTH_SIGMA = 1.0
KP_SMOOTH_RADIUS = 2
KP_PEAK_THRESHOLD = 0.01
KP_PEAK_AMBIGUITY_GAP = 0.25
KP_MISSING_VALUE = -999.999
ADD_THRESHOLDS_MM = np.linspace(0.0, 100.0, 100, dtype=np.float32)
SOURCE_SYNTH = np.uint8(0)
SOURCE_REAL = np.uint8(1)
