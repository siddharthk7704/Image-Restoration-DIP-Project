from .classical import (  # noqa: F401
    denoise_bilateral,
    denoise_gaussian,
    denoise_median,
    denoise_perona_malik,
)

from .nlm import denoise_nlm  # noqa: F401,E402  (library-based)

METHODS = {
    "gaussian": denoise_gaussian,
    "median": denoise_median,
    "bilateral": denoise_bilateral,
    "perona_malik": denoise_perona_malik,
    "nlm": denoise_nlm,
}


def get_filter(name):
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method '{name}'. Available: {sorted(METHODS)}") from None
