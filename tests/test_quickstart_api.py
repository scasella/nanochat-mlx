import importlib

from fastapi.testclient import TestClient


def load_quickstart_module(base_dir, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(base_dir))
    import scripts.quickstart as quickstart

    quickstart = importlib.reload(quickstart)
    quickstart._unload_model()
    return quickstart


def test_quickstart_preflight_errors(tmp_path, monkeypatch):
    quickstart = load_quickstart_module(tmp_path, monkeypatch)
    client = TestClient(quickstart.app)

    status = client.get("/status")
    assert status.status_code == 200
    body = status.json()
    assert body["data"] is False
    assert body["tokenizer"] is False
    assert body["data_shards"] == 0

    tok = client.get("/run/tokenizer")
    assert tok.status_code == 200
    assert "No dataset shards found" in tok.text

    train = client.get("/run/train?depth=4")
    assert train.status_code == 200
    assert "No dataset shards found" in train.text


def test_quickstart_real_status_and_chat(trained_base_dir, monkeypatch):
    quickstart = load_quickstart_module(trained_base_dir, monkeypatch)
    client = TestClient(quickstart.app)

    status = client.get("/status")
    assert status.status_code == 200
    body = status.json()
    assert body["data"] is True
    assert body["tokenizer"] is True
    assert body["train"]["4"] >= 1

    checkpoints = client.get("/checkpoints")
    assert checkpoints.status_code == 200
    checkpoint_list = checkpoints.json()
    assert any(item["depth"] == 4 and item["source"] == "base" for item in checkpoint_list)

    load = client.post("/chat/load", json={"depth": 4, "source": "base"})
    assert load.status_code == 200
    assert load.json()["status"] == "loaded"

    completion = client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Say hello briefly."}],
            "temperature": 0.0,
            "max_tokens": 8,
            "top_k": 1,
        },
    )
    assert completion.status_code == 200
    assert 'data: {"done": true}' in completion.text
    assert 'data: {"token":' in completion.text

    unload = client.post("/chat/unload")
    assert unload.status_code == 200
    assert unload.json()["status"] == "unloaded"
