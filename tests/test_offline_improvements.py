"""
Offline tests (no API keys) for schema voting, retries, document reading and
the OpenIE/ClosedIE wrappers, using a scripted fake provider.

    python -m pytest tests/test_offline_improvements.py -q
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import trallie.utils  # noqa: E402
from trallie import DataExtractor, SchemaGenerator  # noqa: E402
from trallie.data_handlers import DataHandler, DataHandlerError  # noqa: E402
from trallie.providers import BaseProvider, register_provider  # noqa: E402
from trallie.utils import normalize_attribute_name  # noqa: E402


@register_provider("fake")
class FakeProvider(BaseProvider):
    """Returns queued responses in order; records every call."""

    responses = []
    calls = []

    def list_available_models(self):
        return ["fake-model"]

    def do_chat_completion(self, system_prompt, user_prompt, model_name):
        FakeProvider.calls.append(user_prompt)
        return FakeProvider.responses.pop(0)


@pytest.fixture(autouse=True)
def reset_fake(monkeypatch):
    FakeProvider.responses = []
    FakeProvider.calls = []
    monkeypatch.setattr(trallie.utils.time, "sleep", lambda s: None)


def test_normalize_attribute_name():
    assert normalize_attribute_name("intendedUse") == "intended use"
    assert normalize_attribute_name("intended_use") == "intended use"
    assert normalize_attribute_name("Intended  Use") == "intended use"
    assert normalize_attribute_name("510kNumber") == "510k number"
    assert normalize_attribute_name("product-code") == "product code"


def test_schema_votes_merge_name_variants_and_keep_descriptions():
    FakeProvider.responses = [
        json.dumps({"intendedUse": "what the device is for", "applicant": "company"}),
        json.dumps({"intended_use": "other wording", "Applicant": "who applied"}),
        json.dumps({"Intended Use": "x", "panel": "review panel"}),
    ]
    gen = SchemaGenerator(provider="fake", model_name="fake-model")
    names = gen.discover_schema("desc", ["a", "b", "c"], from_text=True, top_k=2)
    assert names == ["intended use", "applicant"]
    assert gen.attribute_counter["intended use"] == 3
    schema = gen.get_top_k_attributes(top_k=3, return_descriptions=True)
    assert schema == {"intended use": "what the device is for", "applicant": "company", "panel": "review panel"}


def test_retry_calls_the_llm_again():
    FakeProvider.responses = ["not json", "", json.dumps({"a": "1"})]
    ext = DataExtractor(provider="fake", model_name="fake-model")
    assert ext.extract_attributes(["a"], "record", max_retries=3) == {"a": "1"}
    assert len(FakeProvider.calls) == 3


def test_think_blocks_are_stripped_without_reasoning_mode():
    FakeProvider.responses = ['<think>reasoning {"no": 1}</think>{"a": "1"}']
    ext = DataExtractor(provider="fake", model_name="fake-model")
    assert ext.extract_attributes(["a"], "record") == {"a": "1"}


def test_schema_with_descriptions_is_sent_as_json():
    FakeProvider.responses = [json.dumps({"a": "1"})]
    DataExtractor(provider="fake", model_name="fake-model").extract_attributes({"a": "the a value"}, "record")
    assert '{"a": "the a value"}' in FakeProvider.calls[0]


def test_data_handler_raises_instead_of_returning_error_text(tmp_path):
    with pytest.raises(DataHandlerError):
        DataHandler(str(tmp_path / "missing.txt")).get_text()
    (tmp_path / "doc.docx").write_text("x")
    with pytest.raises(DataHandlerError):
        DataHandler(str(tmp_path / "doc.docx")).get_text()


def test_html_is_reduced_to_text(tmp_path):
    page = tmp_path / "page.htm"
    page.write_text("<html><head><style>p{}</style><script>var x=1;</script></head>"
                    "<body><h1>Title: Alien</h1><p>Director: <b>Ridley Scott</b></p></body></html>")
    text = DataHandler(str(page)).get_text()
    assert text == "Title: Alien\nDirector:\nRidley Scott"


def test_wrappers_skip_unreadable_records_and_save_schema(tmp_path):
    from trallie.wrappers import closedie, openie

    good = tmp_path / "K1.txt"
    good.write_text("Applicant: ACME")
    missing = str(tmp_path / "K2.txt")
    FakeProvider.responses = [
        json.dumps({"Applicant": "company that applied"}),  # schema discovery on K1 (K2 skipped)
        json.dumps({"applicant": "ACME"}),                  # extraction on K1
    ]
    out = openie("desc", [str(good), missing], "fake", "fake-model", False, "ds", results_dir=str(tmp_path / "r"))
    assert out == {"K1.txt": {"applicant": "ACME"}, "K2.txt": None}
    saved = json.loads((tmp_path / "r" / "fake-model_ds_openie_schema.json").read_text())
    assert saved == {"applicant": "company that applied"}

    FakeProvider.responses = [json.dumps({"applicant": "ACME"})]
    out = closedie([str(good)], ["applicant"], "fake", "fake-model", False, "ds", results_dir=str(tmp_path / "r"))
    assert (tmp_path / "r" / "fake-model_ds_closedie_predicted_table.json").exists()
