from datetime import datetime


def now_timestamp() -> str:
    """Produce a timestamp as a string"""
    time_ = datetime.now()
    stamp = time_.strftime("%Y.%m.%d-%H:%M:%S")
    return stamp
