try:
    from config import MODEL_CONFIGS
except ImportError:
    from ..config import MODEL_CONFIGS


def build_unet(name, in_channels=3, out_channels=1):
    cls, filters = MODEL_CONFIGS[name]
    return cls(in_channels=in_channels, out_channels=out_channels, filters=filters)
