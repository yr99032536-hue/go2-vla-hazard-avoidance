"""Isaac Sim OmniGraph publishers for the simulation-first RTAB-Map slice."""

from __future__ import annotations

from dataclasses import dataclass


CAMERA_FRAME_ID = "camera_optical_frame"
BASE_FRAME_ID = "base_link"
ODOM_FRAME_ID = "odom"
RGB_TOPIC = "camera/color/image_raw"
DEPTH_TOPIC = "camera/depth/image_rect_raw"
CAMERA_INFO_TOPIC = "camera/camera_info"
WRIST_CAMERA_FRAME_ID = "wrist_camera_optical_frame"
WRIST_RGB_TOPIC = "wrist_camera/color/image_raw"
WRIST_DEPTH_TOPIC = "wrist_camera/depth/image_rect_raw"
WRIST_CAMERA_INFO_TOPIC = "wrist_camera/camera_info"
ODOM_TOPIC = "/odom"
TF_TOPIC = "/tf"
CLOCK_TOPIC = "/clock"
CAMERA_FRAME_SKIP = 5


@dataclass
class Ros2SlamBridgeHandles:
    """Objects that must remain alive while OmniGraph publishes."""

    camera_viewport: object
    wrist_camera_viewport: object | None = None


def enable_ros2_bridge_extension(simulation_app) -> None:
    """Enable Isaac Sim's bundled Humble bridge before graph construction."""
    from isaacsim.core.utils import extensions

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()


def _graph_spec(graph_path: str) -> dict:
    import omni.graph.core as og

    return {
        "graph_path": graph_path,
        "evaluator_name": "execution",
        "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
    }


def setup_camera_and_clock_graph(
    camera_prim_path: str,
    width: int,
    height: int,
) -> Ros2SlamBridgeHandles:
    """Publish front RGB, depth, camera info, and simulation clock."""
    import omni.graph.core as og
    from omni.kit.viewport.utility import create_viewport_window

    viewport = create_viewport_window(
        "Active SLAM ROS2 Camera",
        width=width,
        height=height,
        visible=False,
    )
    viewport.viewport_api.set_active_camera(camera_prim_path)
    viewport.viewport_api.set_texture_resolution((width, height))
    render_product_path = viewport.viewport_api.get_render_product_path()
    if not render_product_path:
        raise RuntimeError("ROS2 camera render product was not created")

    keys = og.Controller.Keys
    og.Controller.edit(
        _graph_spec("/ActiveSlam/ROS2Camera"),
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "rgb.inputs:execIn"),
                ("tick.outputs:tick", "depth.inputs:execIn"),
                ("tick.outputs:tick", "info.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("rgb.inputs:renderProductPath", render_product_path),
                ("rgb.inputs:frameId", CAMERA_FRAME_ID),
                ("rgb.inputs:topicName", RGB_TOPIC),
                ("rgb.inputs:type", "rgb"),
                ("rgb.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                ("depth.inputs:renderProductPath", render_product_path),
                ("depth.inputs:frameId", CAMERA_FRAME_ID),
                ("depth.inputs:topicName", DEPTH_TOPIC),
                ("depth.inputs:type", "depth"),
                ("depth.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                ("info.inputs:renderProductPath", render_product_path),
                ("info.inputs:frameId", CAMERA_FRAME_ID),
                ("info.inputs:topicName", CAMERA_INFO_TOPIC),
                ("info.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
            ],
        },
    )
    og.Controller.edit(
        _graph_spec("/ActiveSlam/ROS2Clock"),
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("time.outputs:simulationTime", "clock.inputs:timeStamp"),
            ],
            keys.SET_VALUES: [("clock.inputs:topicName", CLOCK_TOPIC)],
        },
    )
    return Ros2SlamBridgeHandles(camera_viewport=viewport)


def setup_odometry_graph(chassis_prim_path: str) -> None:
    """Publish ground-truth odometry and odom→base_link TF from one tick."""
    import omni.graph.core as og
    import omni.usd
    from pxr import Sdf

    keys = og.Controller.Keys
    og.Controller.edit(
        _graph_spec("/ActiveSlam/ROS2Odom"),
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("compute", "isaacsim.core.nodes.IsaacComputeOdometry"),
                ("odom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ("tf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "compute.inputs:execIn"),
                ("compute.outputs:execOut", "odom.inputs:execIn"),
                ("compute.outputs:execOut", "tf.inputs:execIn"),
                ("time.outputs:simulationTime", "odom.inputs:timeStamp"),
                ("time.outputs:simulationTime", "tf.inputs:timeStamp"),
                ("compute.outputs:position", "odom.inputs:position"),
                ("compute.outputs:orientation", "odom.inputs:orientation"),
                ("compute.outputs:linearVelocity", "odom.inputs:linearVelocity"),
                ("compute.outputs:angularVelocity", "odom.inputs:angularVelocity"),
                ("compute.outputs:position", "tf.inputs:translation"),
                ("compute.outputs:orientation", "tf.inputs:rotation"),
            ],
            keys.SET_VALUES: [
                ("odom.inputs:chassisFrameId", BASE_FRAME_ID),
                ("odom.inputs:odomFrameId", ODOM_FRAME_ID),
                ("odom.inputs:topicName", ODOM_TOPIC),
                ("tf.inputs:parentFrameId", ODOM_FRAME_ID),
                ("tf.inputs:childFrameId", BASE_FRAME_ID),
                ("tf.inputs:topicName", TF_TOPIC),
            ],
        },
    )
    stage = omni.usd.get_context().get_stage()
    compute_prim = stage.GetPrimAtPath("/ActiveSlam/ROS2Odom/compute")
    if not compute_prim.IsValid():
        raise RuntimeError("ROS2 odometry compute prim was not created")
    compute_prim.GetRelationship("inputs:chassisPrim").SetTargets([Sdf.Path(chassis_prim_path)])


def setup_wrist_camera_graph(camera_prim_path: str, width: int, height: int) -> object:
    """Publish wrist RGB-D and CameraInfo on canonical, separately framed topics."""
    import omni.graph.core as og
    from omni.kit.viewport.utility import create_viewport_window

    viewport = create_viewport_window(
        "Active SLAM ROS2 Wrist Camera",
        width=width,
        height=height,
        visible=False,
    )
    viewport.viewport_api.set_active_camera(camera_prim_path)
    viewport.viewport_api.set_texture_resolution((width, height))
    render_product_path = viewport.viewport_api.get_render_product_path()
    if not render_product_path:
        raise RuntimeError("ROS2 wrist camera render product was not created")
    keys = og.Controller.Keys
    og.Controller.edit(
        _graph_spec("/ActiveSlam/ROS2WristCamera"),
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "rgb.inputs:execIn"),
                ("tick.outputs:tick", "depth.inputs:execIn"),
                ("tick.outputs:tick", "info.inputs:execIn"),
            ],
            keys.SET_VALUES: [
                ("rgb.inputs:renderProductPath", render_product_path),
                ("rgb.inputs:frameId", WRIST_CAMERA_FRAME_ID),
                ("rgb.inputs:topicName", WRIST_RGB_TOPIC),
                ("rgb.inputs:type", "rgb"),
                ("rgb.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                ("depth.inputs:renderProductPath", render_product_path),
                ("depth.inputs:frameId", WRIST_CAMERA_FRAME_ID),
                ("depth.inputs:topicName", WRIST_DEPTH_TOPIC),
                ("depth.inputs:type", "depth"),
                ("depth.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
                ("info.inputs:renderProductPath", render_product_path),
                ("info.inputs:frameId", WRIST_CAMERA_FRAME_ID),
                ("info.inputs:topicName", WRIST_CAMERA_INFO_TOPIC),
                ("info.inputs:frameSkipCount", CAMERA_FRAME_SKIP),
            ],
        },
    )
    return viewport


def setup_slam_publishers(
    simulation_app,
    camera_prim_path: str,
    wrist_camera_prim_path: str | None,
    chassis_prim_path: str,
    width: int,
    height: int,
    wrist_width: int,
    wrist_height: int,
) -> Ros2SlamBridgeHandles:
    """Enable and install the complete read-only SLAM publisher graph."""
    enable_ros2_bridge_extension(simulation_app)
    handles = setup_camera_and_clock_graph(camera_prim_path, width, height)
    if wrist_camera_prim_path is not None:
        handles.wrist_camera_viewport = setup_wrist_camera_graph(
            wrist_camera_prim_path,
            wrist_width,
            wrist_height,
        )
    setup_odometry_graph(chassis_prim_path)
    return handles
