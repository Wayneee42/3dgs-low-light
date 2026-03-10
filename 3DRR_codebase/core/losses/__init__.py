def build_loss_modules(*args, **kwargs):
    from .builder import build_loss_modules as _build_loss_modules

    return _build_loss_modules(*args, **kwargs)



def compute_loss_modules(*args, **kwargs):
    from .builder import compute_loss_modules as _compute_loss_modules

    return _compute_loss_modules(*args, **kwargs)



def requires_depth_render(*args, **kwargs):
    from .builder import requires_depth_render as _requires_depth_render

    return _requires_depth_render(*args, **kwargs)


__all__ = ["build_loss_modules", "compute_loss_modules", "requires_depth_render"]
