"""Canonical QoS factories for P1A active-SLAM interfaces."""

from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

SENSOR_CAPTURE_DEPTH = 2
SNAPSHOT_DEPTH = 1
SNAPSHOT_LIFESPAN_MS = 500
ACTION_DEPTH = 1
APPLICATION_ACK_DEPTH = 32
PROPOSAL_FEEDBACK_MAX_HZ = 4
APPLY_TRAJECTORY_FEEDBACK_MAX_HZ = 20


def sensor_capture_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=SENSOR_CAPTURE_DEPTH,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def snapshot_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=SNAPSHOT_DEPTH,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
        lifespan=Duration(nanoseconds=SNAPSHOT_LIFESPAN_MS * 1_000_000),
    )


def action_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=ACTION_DEPTH,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def application_ack_qos() -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=APPLICATION_ACK_DEPTH,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
