from __future__ import annotations


def test_local_storage_roundtrip(local_storage):
    key = local_storage.run_key("run_x", "inputs", "foo.json")
    local_storage.write_json(key, {"hello": "world"})
    assert local_storage.exists(key)
    assert local_storage.read_json(key) == {"hello": "world"}
    assert key in local_storage.list_prefix(local_storage.run_key("run_x"))
