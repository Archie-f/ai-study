def is_span_in_one_window(span_start: int, span_length: int,
                          n: int, o: int) -> bool:
    """Return True if words span_start .. span_start + span_length - 1
    all fall inside at least one window.

    Examples (n=10, o=3 -> windows start at 0, 7, 14, ...):
        (5, 4, 10, 3)  -> True   words 5-8; window 0 covers 0-9
        (5, 6, 10, 3)  -> False  words 5-10 are split
        (8, 9, 10, 3)  -> True   words 8-16; window 7 covers 7-16
        (0, 11, 10, 3) -> False  longer than any window
    """
    span = (span_start, (span_start + span_length - 1))
    overlapped_length = n - o
    start_point = (span_start // overlapped_length) * overlapped_length
    end_point = start_point + (n - 1)
    window = (start_point, end_point)

    return window[0] <= span[0] and span[1] <= window[1]
