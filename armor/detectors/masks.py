def _mask_last4(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


# DLP framework doc section 4: mask_fn renders evidence without the raw value.
MASKS = {"last4": _mask_last4}
