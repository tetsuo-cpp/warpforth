"""Unit tests for concurrent-safe Vast.ai session ownership."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import requests

from gpu_test import conftest as gpu_conftest
from gpu_test.conftest import VastSession

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeVastAI:
    """Small in-memory substitute for the Vast.ai lifecycle API."""

    def __init__(self, instances: list[dict[str, object]] | None = None) -> None:
        self.instances = instances or []
        self.destroyed: list[int] = []
        self.next_id = 100
        self.create_error: Exception | None = None
        self.rejected_offers: set[int] = set()
        self.destroy_failures_before_remove = 0
        self.create_calls: list[dict[str, object]] = []
        self.constructor_kwargs: dict[str, object] = {}
        self.hide_after_create = False
        self.hide_next_listing = False

    def show_instances(self) -> list[dict[str, object]]:
        if self.hide_next_listing:
            self.hide_next_listing = False
            return []
        return [dict(instance) for instance in self.instances]

    def search_offers(self, **_kwargs: object) -> list[dict[str, int]]:
        return [{"id": 1}, {"id": 2}]

    def create_instance(self, *, label: str, **kwargs: object) -> dict[str, object]:
        self.create_calls.append({"label": label, **kwargs})
        offer_id = int(str(kwargs["id"]))
        if offer_id in self.rejected_offers:
            return {"success": False}

        instance_id = self.next_id
        self.next_id += 1
        self.instances.append(
            {
                "id": instance_id,
                "label": label,
                "actual_status": "loading",
                "dph_total": 0.2,
            }
        )
        self.hide_next_listing = self.hide_after_create
        if self.create_error is not None:
            raise self.create_error
        return {"success": True, "new_contract": instance_id}

    def destroy_instance(self, **kwargs: object) -> dict[str, bool]:
        instance_id = int(str(kwargs["id"]))
        self.destroyed.append(instance_id)
        if self.destroy_failures_before_remove:
            self.destroy_failures_before_remove -= 1
            return {"success": True}
        self.instances = [
            instance for instance in self.instances if instance.get("id") != instance_id
        ]
        return {"success": True}


def configure_session_test(
    monkeypatch: pytest.MonkeyPatch,
    sdk: FakeVastAI,
    wait_for_ssh: Callable[[VastSession], None] | None = None,
) -> None:
    """Install deterministic SDK, sleep, and SSH behavior for one test."""

    def no_wait(_session: VastSession) -> None:
        return

    def create_sdk(_api_key: str, **kwargs: object) -> FakeVastAI:
        sdk.constructor_kwargs = kwargs
        return sdk

    monkeypatch.setattr(gpu_conftest, "VastAI", create_sdk)
    monkeypatch.setattr(gpu_conftest.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(VastSession, "_wait_for_ssh", wait_for_ssh or no_wait)


def test_concurrent_sessions_destroy_only_their_own_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeVastAI()
    configure_session_test(monkeypatch, sdk)

    with VastSession("key") as first:
        with VastSession("key") as second:
            assert first.instance_label != second.instance_label
            assert {instance["id"] for instance in sdk.instances} == {100, 101}
        assert {instance["id"] for instance in sdk.instances} == {100}

    assert sdk.instances == []
    assert sdk.destroyed == [101, 100]


def test_startup_logs_existing_instances_without_destroying_them(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sdk = FakeVastAI(
        [
            {
                "id": 40,
                "label": "warpforth-test-existing-run",
                "actual_status": "running",
                "gpu_name": "RTX 4090",
                "dph_total": 0.25,
            },
            {"id": 41, "label": "unrelated", "actual_status": "running", "dph_total": 1.0},
        ]
    )
    configure_session_test(monkeypatch, sdk)

    with caplog.at_level(logging.INFO), VastSession("key"):
        assert {instance["id"] for instance in sdk.instances} == {40, 41, 100}

    assert {instance["id"] for instance in sdk.instances} == {40, 41}
    assert sdk.destroyed == [100]
    assert "id=40 label=warpforth-test-existing-run" in caplog.text
    assert "total known cost=$0.250/hr" in caplog.text
    assert "id=41" not in caplog.text
    assert "will not be destroyed" in caplog.text


def test_definitively_rejected_offer_tries_the_next_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeVastAI()
    sdk.rejected_offers.add(1)
    configure_session_test(monkeypatch, sdk)

    with VastSession("key") as session:
        assert session.instance_id == 100
        assert sdk.create_calls[1]["runtype"] == "ssh_direc ssh_proxy"
        assert sdk.constructor_kwargs == {"retry": 1}

    assert sdk.destroyed == [100]


def test_ambiguous_create_failure_reconciles_by_unique_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeVastAI()
    sdk.create_error = requests.ConnectionError("response lost")
    configure_session_test(monkeypatch, sdk)

    with VastSession("key") as session:
        assert session.instance_id == 100

    assert sdk.destroyed == [100]
    assert len(sdk.create_calls) == 1


def test_setup_interrupt_destroys_the_owned_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeVastAI()
    sdk.hide_after_create = True

    def interrupt(_session: VastSession) -> None:
        raise KeyboardInterrupt

    configure_session_test(monkeypatch, sdk, interrupt)

    with pytest.raises(KeyboardInterrupt), VastSession("key"):
        pytest.fail("session setup unexpectedly completed")

    assert sdk.instances == []
    assert sdk.destroyed == [100]


def test_cleanup_verifies_instance_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = FakeVastAI()
    sdk.destroy_failures_before_remove = 2
    configure_session_test(monkeypatch, sdk)

    with VastSession("key"):
        pass

    assert sdk.instances == []
    assert sdk.destroyed == [100, 100, 100]


def test_duplicate_unique_labels_fail_without_adopting_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = FakeVastAI()
    configure_session_test(monkeypatch, sdk)
    session = VastSession("key")

    def create_duplicate(**kwargs: object) -> dict[str, object]:
        label = str(kwargs["label"])
        sdk.instances.extend([{"id": 200, "label": label}, {"id": 201, "label": label}])
        message = "response lost"
        raise requests.ConnectionError(message)

    monkeypatch.setattr(sdk, "create_instance", create_duplicate)

    with pytest.raises(RuntimeError, match="Multiple instances found for unique label"), session:
        pytest.fail("duplicate labels unexpectedly completed setup")

    assert sdk.instances == []
    assert sdk.destroyed == [200, 201]
