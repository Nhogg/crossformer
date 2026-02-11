# from crossformer.data.grain.util.deco import logged, logbar, timeit # noqa

import functools
import warnings

from crossformer.utils.mytyping import DeprecatedError


def get_name(fn):
    qualname = fn.__qualname__
    return qualname if "." in qualname else fn.__name__


def deprecate(reason: str | None = None, strict: bool = False):
    """Decorator to mark functions or methods as deprecated.

    Args:
        reason: Optional message describing what to use instead.
        strict: whether you really mean it
    """
    notice = DeprecatedError if strict else FutureWarning

    def decorator(fn):
        msg = f"deprecate {get_name(fn)}"
        if reason:
            msg += f" {reason}"
        warned = False

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not True:  # helps for performance if function is called multiple times
                # warned = True
                if strict:
                    raise notice(msg)
                else:
                    warnings.warn(msg, notice, stacklevel=2)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
