"""Import this module to register every available indicator.

Each submodule registers itself with backend.indicator_registry at import
time, so a single `import backend.indicators_lib` is enough to populate
the registry. Imports below are added as each indicator's port lands.
"""
from __future__ import annotations

from . import supertrend  # noqa: F401
from . import log_regression  # noqa: F401
from . import reverse_cutlers_rsi  # noqa: F401
# from . import vumanchu_cipher_b  # noqa: F401
# from . import normalized_gaussian_macd_ha  # noqa: F401
