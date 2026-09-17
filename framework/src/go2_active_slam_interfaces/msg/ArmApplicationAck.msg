# Durable application acknowledgment published only by the Isaac action server.
uint8 RESULT_COMPLETED=0
uint8 RESULT_REJECTED_PRESTART=1
uint8 RESULT_ABORTED_STALE=2
uint8 RESULT_ABORTED_CONTACT=3
uint8 RESULT_ABORTED_DIVERGENCE=4
uint8 RESULT_CANCELED=5
uint8 RESULT_INTERNAL_ERROR=6

uint8 MODE_DISABLED=0
uint8 MODE_ORACLE=1
uint8 MODE_SHADOW=2
uint8 MODE_LEARNED=3

uint8 SOURCE_ORACLE=0
uint8 SOURCE_LEARNED=1
uint8 SOURCE_HUMAN=2
uint8 SOURCE_HOME=3

uint16 schema_version
uint8 result_code
uint8[16] ack_uuid
uint8[16] application_uuid
uint8[16] authorization_uuid
uint8[16] request_uuid
uint8[16] snapshot_uuid
uint64 episode_id
uint64 decision_epoch
uint64 map_revision
uint64 target_revision
uint8 mode
uint8 proposal_source
bool authorization_consumed
int64 accepted_at_ns
int64 started_at_ns
int64 finished_at_ns
uint16 completed_waypoints
float32[7] authorized_target_external_deg
float32[7] actual_final_external_deg
uint8[32] applied_path_sha256
uint8[32] result_payload_sha256
string<=128 reason
