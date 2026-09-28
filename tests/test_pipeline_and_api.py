import io

import numpy as np
import pytest
from PIL import Image

from rarecare.app import create_app
from rarecare.config import AppConfig, Config
from rarecare.schemas import ComponentMode, Tier


def test_pipeline_end_to_end_with_question(pipe):
    session, r = pipe.start("I keep bleeding from down there and have safed pani for 2 months")
    assert r.tier is Tier.NEED_MORE_INFO and r.question.variable == "bleeding_context"
    r = pipe.answer(session, "bleeding_context", "after_menopause")
    assert r.tier is Tier.URGENT
    assert {e.criterion_id for e in r.evidence} >= {"CX-DISCHARGE"}
    assert any(e.criterion_id.startswith("EN-PMB") for e in r.evidence)
    assert r.component_modes["image_models"] is ComponentMode.UNAVAILABLE
    assert r.conformal_calibrated is False


def test_answering_twice_is_rejected(pipe):
    session, _ = pipe.start("lump in my breast")
    pipe.answer(session, "age_band", "30-49")
    with pytest.raises(ValueError):
        pipe.answer(session, "age_band", "55+")


def test_questionnaire_only_input(pipe):
    _, r = pipe.start("", age_band="55+", selected_concepts=["postmenopausal_bleeding"])
    assert r.tier is Tier.URGENT
    assert r.evidence[0].from_questionnaire


def test_questionnaire_rejects_unknown_symptom(pipe):
    with pytest.raises(ValueError):
        pipe.start("", selected_concepts=["not_a_symptom"])


def test_negated_only_is_watch(pipe):
    _, r = pipe.start("no lump in my breast and no bleeding after sex")
    assert r.tier is Tier.WATCH and r.evidence == []


@pytest.fixture()
def client(pipe):
    cfg = Config(app=AppConfig(rate_limit_per_minute=1000))
    return create_app(cfg, pipe).test_client()


def test_health_and_headers(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert r.headers["Cache-Control"] == "no-store"


def test_index_is_discreet(client):
    html = client.get("/").get_data(as_text=True)
    assert "<title>Notes</title>" in html


def test_assess_validation(client):
    assert client.post("/api/v1/assess", json={"text": ""}).status_code == 422
    assert client.post("/api/v1/assess", json={"text": "x" * 5000}).status_code == 422
    assert client.post("/api/v1/assess", json={"text": "lump", "age_band": "old"}).status_code == 422
    assert client.post("/api/v1/assess", data="nope", content_type="text/plain").status_code == 400


def test_session_flow_and_expiry(client):
    r = client.post("/api/v1/assess", json={"text": "there is a lump in my breast"}).json
    sid = r["session_id"]
    assert r["result"]["question"]["variable"] == "age_band"
    r2 = client.post(f"/api/v1/session/{sid}/answer", json={"variable": "age_band", "value": "30-49"}).json
    assert r2["result"]["tier"] == "URGENT" and r2["session_id"] is None
    # Session is deleted once the decision is final
    assert (
        client.post(f"/api/v1/session/{sid}/answer", json={"variable": "age_band", "value": "30-49"}).status_code == 404
    )


def test_bad_answer_is_422(client):
    sid = client.post("/api/v1/assess", json={"text": "lump in my breast"}).json["session_id"]
    assert (
        client.post(f"/api/v1/session/{sid}/answer", json={"variable": "age_band", "value": "999"}).status_code == 422
    )


def test_multipart_with_image(client):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(40, 200, (256, 256, 3)).astype(np.uint8)).save(buf, "PNG")
    r = client.post(
        "/api/v1/assess",
        data={"text": "lump in my breast", "age_band": "30-49", "image": (io.BytesIO(buf.getvalue()), "scan.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert r.json["result"]["image"]["status"] == "MODEL_UNAVAILABLE"


def test_invalid_image_is_422(client):
    r = client.post(
        "/api/v1/assess",
        data={"text": "lump", "image": (io.BytesIO(b"garbage"), "x.png")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 422 and r.json["error"]["code"] == "invalid_image"


def test_rate_limit(pipe):
    c = create_app(Config(app=AppConfig(rate_limit_per_minute=2)), pipe).test_client()
    codes = [c.post("/api/v1/assess", json={"text": "lump"}).status_code for _ in range(3)]
    assert codes[-1] == 429


def test_questionnaire_endpoint(client):
    q = client.get("/api/v1/questionnaire").json
    assert {a["id"] for a in q["areas"]} == {"breast", "gynaecological", "abdominal"}
    assert q["age_bands"][0] == "<30"
