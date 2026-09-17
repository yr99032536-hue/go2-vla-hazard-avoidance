from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
import struct

TX = Path("/home/iy/Isaac/Robotics/robot_models/soarm_nbv/contracts/transaction_v2.json")
GOLDENS = {
    "snapshot": "91acdbf977f356a42dea419d4fff2401e6057013aaf9181d5ecb0459fb6fa8c2",
    "proposal": "d56f2e788150baaf033705d4ec4a4eafdab44b5776e89576433a35ade2e13b37",
    "path_hash_preimage": "7dde950123fe44884b5db4a34a55295ad658b87927bfab301153075847526b91",
    "apply_goal_hmac_preimage": "309baafae6a543b19bd58ebf67227b98a23b637dd88716732840c2137f7e7976",
    "result": "33abf9e059090b38848de1bf4bc7717622dff1ad6412a45131d9f8d4ee77f865",
    "ack": "582ed560da9ea18bcec1960ca41951c9cda98614ee524f270f7293d86d4c3d26",
    "source_manifest": "bfbb67562f7350235f904f0c94fd238afd9606aeb2fce290c71b739582af5310",
    "checkpoint_allowlist": "408e2f9ab39abb53b02ca5471b8bcecba9997e7f92025bd2699dbc5dc768dc3b",
}
HMAC_GOLDEN = "3ede519bb21ca14677429cd181e1e5710469815ecebff9302af0a9f3ff9b6657"


def payload(kind: str, field_id: int) -> bytes:
    if kind == "uuid": return bytes([field_id]) * 16
    if kind == "digest": return bytes([field_id]) * 32
    if kind == "uint16": return struct.pack("<H", field_id)
    if kind == "uint32": return struct.pack("<I", field_id)
    if kind == "uint64": return struct.pack("<Q", field_id)
    if kind == "int64": return struct.pack("<q", field_id)
    if kind == "float32": return struct.pack("<f", float(field_id))
    if kind == "float32[7]": return struct.pack("<7f", *(field_id + n for n in range(7)))
    if kind == "bool" or kind.startswith("enum("): return b"\0"
    if kind.startswith("bytes["): return bytes([field_id]) * int(kind[6:-1])
    if kind.startswith("string<="): return b"\x02\x00ok"
    if kind == "points": return struct.pack("<Hq21f", 1, 5_000_000, *range(16, 23), *(0.0,) * 14)
    if kind == "digest_sequence_sorted<=65535": return struct.pack("<H", 2) + b"a" * 32 + b"b" * 32
    raise AssertionError(kind)


def encode(domain: dict, fields: list[tuple[int, bytes]]) -> bytes:
    declared = [field[0] for field in domain["fields"]]
    received = [field_id for field_id, _ in fields]
    if received != declared or len(received) != len(set(received)):
        raise ValueError("non-canonical field inventory")
    return b"ASCV2" + struct.pack("<HHH", domain["id"], 2, len(fields)) + b"".join(
        struct.pack("<HI", field_id, len(value)) + value for field_id, value in fields
    )


def validate(kind: str, value: bytes, enums: dict) -> None:
    fixed = {"uuid": 16, "digest": 32, "uint16": 2, "uint32": 4, "uint64": 8,
             "int64": 8, "float32": 4, "float32[7]": 28, "bool": 1}
    if kind in fixed and len(value) != fixed[kind]: raise ValueError("length")
    if kind == "bool" and value not in (b"\0", b"\1"): raise ValueError("bool")
    if kind.startswith("enum("):
        enum_name = kind[5:-1]
        if len(value) != 1 or value[0] not in enums[enum_name].values(): raise ValueError("enum")
    if kind.startswith("float32"):
        values = struct.unpack("<" + "f" * (len(value) // 4), value)
        if not all(math.isfinite(item) for item in values): raise ValueError("nonfinite")
    if kind.startswith("bytes[") and len(value) != int(kind[6:-1]): raise ValueError("bytes")
    if kind.startswith("string<="):
        if len(value) < 2: raise ValueError("string")
        size = struct.unpack("<H", value[:2])[0]
        if len(value) != 2 + size or size > int(kind[8:]): raise ValueError("string")
        value[2:].decode("utf-8")
    if kind == "digest_sequence_sorted<=65535":
        count = struct.unpack("<H", value[:2])[0] if len(value) >= 2 else 0
        items = [value[2 + 32 * index:34 + 32 * index] for index in range(count)]
        if not 1 <= count <= 65535 or len(value) != 2 + 32 * count or items != sorted(set(items)): raise ValueError("allowlist")
    if kind == "points":
        count = struct.unpack("<H", value[:2])[0] if len(value) >= 2 else 0
        if not 1 <= count <= 128 or len(value) != 2 + 92 * count: raise ValueError("points")
        prior = 0
        for offset in range(2, len(value), 92):
            time_ns, *vectors = struct.unpack_from("<q21f", value, offset)
            if not 0 < time_ns <= 5_000_000_000 or time_ns <= prior: raise ValueError("time")
            if not all(math.isfinite(item) for item in vectors): raise ValueError("vector")
            prior = time_ns


def decode(domain: dict, record: bytes, enums: dict) -> dict[str, bytes]:
    if len(record) < 11 or record[:5] != b"ASCV2": raise ValueError("envelope")
    domain_id, version, count = struct.unpack_from("<HHH", record, 5)
    if (domain_id, version, count) != (domain["id"], 2, len(domain["fields"])): raise ValueError("header")
    position, values = 11, {}
    for field_id, name, kind in domain["fields"]:
        if position + 6 > len(record): raise ValueError("omitted")
        received_id, size = struct.unpack_from("<HI", record, position)
        position += 6
        if received_id != field_id or position + size > len(record): raise ValueError("field")
        value = record[position:position + size]
        position += size
        validate(kind, value, enums)
        values[name] = value
    if position != len(record): raise ValueError("trailing")
    return values


def fixtures(domains: dict) -> dict[str, bytes]:
    def simple(name: str) -> bytes:
        return encode(domains[name], [(field_id, payload(kind, field_id)) for field_id, _, kind in domains[name]["fields"]])

    path_preimage = simple("path_hash_preimage")
    path_hash = hashlib.sha256(path_preimage).digest()
    goal_fields = [
        (field_id, path_hash if name == "path_sha256" else payload(kind, field_id))
        for field_id, name, kind in domains["apply_goal_hmac_preimage"]["fields"]
    ]
    goal = encode(domains["apply_goal_hmac_preimage"], goal_fields)
    target = dict(goal_fields)[16]
    result_fields = [
        (field_id, target if name == "authorized_target_external_deg" else path_hash if name == "applied_path_sha256" else payload(kind, field_id))
        for field_id, name, kind in domains["result"]["fields"]
    ]
    result = encode(domains["result"], result_fields)
    result_values = {name: value for (_, name, _), (_, value) in zip(domains["result"]["fields"], result_fields)}
    result_hash = hashlib.sha256(result).digest()
    ack_fields = [
        (field_id, result_hash if name == "result_payload_sha256" else result_values.get(name, payload(kind, field_id)))
        for field_id, name, kind in domains["ack"]["fields"]
    ]
    return {
        "snapshot": simple("snapshot"), "proposal": simple("proposal"),
        "path_hash_preimage": path_preimage, "apply_goal_hmac_preimage": goal,
        "result": result, "ack": encode(domains["ack"], ack_fields),
        "source_manifest": simple("source_manifest"),
        "checkpoint_allowlist": simple("checkpoint_allowlist"),
    }


def assert_result_ack(result: bytes, ack: bytes, goal: bytes, domains: dict, enums: dict) -> None:
    result_values = decode(domains["result"], result, enums)
    ack_values = decode(domains["ack"], ack, enums)
    goal_values = decode(domains["apply_goal_hmac_preimage"], goal, enums)
    shared = set(result_values) & set(ack_values)
    if any(result_values[name] != ack_values[name] for name in shared): raise ValueError("result/ack")
    if ack_values["result_payload_sha256"] != hashlib.sha256(result).digest(): raise ValueError("result hash")
    if result_values["applied_path_sha256"] != goal_values["path_sha256"]: raise ValueError("path hash")
    if result_values["authorized_target_external_deg"] != goal_values["validated_target_external_deg"]: raise ValueError("target")
    if goal_values["points"][-84:-56] != goal_values["validated_target_external_deg"]: raise ValueError("endpoint")


def test_complete_canonical_goldens() -> None:
    contract = json.loads(TX.read_text(encoding="utf-8"))
    domains = contract["canonical_serialization_v2"]["domains"]
    built = fixtures(domains)
    for name, record in built.items():
        assert decode(domains[name], record, contract["idl_equivalent"]["enums"])
        assert hashlib.sha256(record).hexdigest() == GOLDENS[name]
    assert hmac.new(bytes(range(32)), built["apply_goal_hmac_preimage"], hashlib.sha256).hexdigest() == HMAC_GOLDEN
    assert_result_ack(built["result"], built["ack"], built["apply_goal_hmac_preimage"], domains, contract["idl_equivalent"]["enums"])


def test_malformed_endpoint_and_same_wrong_path_are_rejected() -> None:
    contract = json.loads(TX.read_text(encoding="utf-8"))
    domains, enums = contract["canonical_serialization_v2"]["domains"], contract["idl_equivalent"]["enums"]
    built = fixtures(domains)
    for malformed in (built["ack"][:-1], built["ack"] + b"x"):
        try: decode(domains["ack"], malformed, enums)
        except ValueError: pass
        else: raise AssertionError("malformed canonical ack accepted")

    wrong_path = b"\x99" * 32
    result_values = decode(domains["result"], built["result"], enums)
    wrong_result = encode(domains["result"], [
        (field_id, wrong_path if name == "applied_path_sha256" else result_values[name])
        for field_id, name, _ in domains["result"]["fields"]
    ])
    ack_values = decode(domains["ack"], built["ack"], enums)
    wrong_ack = encode(domains["ack"], [
        (field_id, wrong_path if name == "applied_path_sha256" else hashlib.sha256(wrong_result).digest() if name == "result_payload_sha256" else ack_values[name])
        for field_id, name, _ in domains["ack"]["fields"]
    ])
    try: assert_result_ack(wrong_result, wrong_ack, built["apply_goal_hmac_preimage"], domains, enums)
    except ValueError: pass
    else: raise AssertionError("result and ack accepted an unauthorized shared path hash")
