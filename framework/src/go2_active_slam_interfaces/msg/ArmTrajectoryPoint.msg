# Exact external seven-motor trajectory sample.
# time_from_start_ns must be strictly increasing, >0, and <=5000000000.
int64 time_from_start_ns
float32[7] position_external_deg
float32[7] velocity_external_deg_s
float32[7] acceleration_external_deg_s2
