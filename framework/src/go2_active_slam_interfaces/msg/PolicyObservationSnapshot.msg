# Immutable, settled observation published by the supervisor.
uint8 MODE_DISABLED=0
uint8 MODE_ORACLE=1
uint8 MODE_SHADOW=2
uint8 MODE_LEARNED=3

uint16 schema_version
uint8[16] snapshot_uuid
uint8[16] frame_authorization_uuid
uint64 episode_id
uint64 decision_epoch
uint64 map_revision
uint64 target_revision
int64 snapshot_stamp_ns
int64 wrist_stamp_ns
int64 front_stamp_ns
int64 state_stamp_ns
int64 guidance_stamp_ns
int64 expires_at_ns
uint8 mode
uint8[32] renderer_sha256
uint8[32] camera_info_sha256
uint8[32] snapshot_payload_sha256
uint8[230400] camera1_wrist_rgb
uint8[921600] camera2_front_rgb
uint8[921600] camera3_guidance_rgb
float32[7] state_external_deg
string<=256 task
