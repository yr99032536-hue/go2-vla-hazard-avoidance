"""Bounded open-floor command response probe; no planner or position writes."""
SEGMENTS = (
    ('initial_stand', 3., (0., 0., 0.)),
    ('forward_slow', 6., (.15, 0., 0.)),
    ('stop_1', 3., (0., 0., 0.)),
    ('lateral_slow', 6., (0., -.15, 0.)),
    ('stop_2', 3., (0., 0., 0.)),
    ('forward', 6., (.30, 0., 0.)),
    ('stop_3', 3., (0., 0., 0.)),
    ('lateral', 6., (0., -.30, 0.)),
    ('stop_4', 3., (0., 0., 0.)),
    ('curve', 6., (.20, 0., .30)),
    ('stop_5', 3., (0., 0., 0.)),
    ('mixed', 6., (.15, -.12, .30)),
    ('stop_6', 3., (0., 0., 0.)),
    ('reverse_slow', 6., (-.08, 0., 0.)),
    ('final_stand', 4., (0., 0., 0.)),
)
DURATION = sum(segment[1] for segment in SEGMENTS)


def command_at(seconds):
    if seconds < 0:
        raise ValueError('Probe time must be nonnegative')
    elapsed = 0.
    for label, duration, command in SEGMENTS:
        if seconds < elapsed + duration:
            return label, command
        elapsed += duration
    return 'done', (0., 0., 0.)
