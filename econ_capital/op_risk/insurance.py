def apply_mitigation(losses, limit=None, deductible=0, coverage=None):  # pylint: disable=unused-argument
    """Apply insurance limit and deductible per loss."""
    capped = [max(0, min(loss - deductible, limit or loss)) for loss in losses]
    return capped
