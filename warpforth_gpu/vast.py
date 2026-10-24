"""Rent one GPU instance on Vast.ai for the lifetime of a context manager."""

from __future__ import annotations

import atexit
import logging
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from requests import RequestException
from vastai import VastAI

if TYPE_CHECKING:
    from typing import Self

logger = logging.getLogger(__name__)

MAX_COST_PER_HOUR = 0.50
CUDA_VERSION = "12.4"
CUDA_IMAGE = f"nvidia/cuda:{CUDA_VERSION}.0-devel-ubuntu22.04"
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 300
INSTANCE_LABEL_PREFIX = "warpforth-test-"
DEFAULT_ARCHITECTURE = "sm_70"


class VastSession:
    """Context manager that rents a GPU instance on Vast.ai.

    Handles searching for offers, launching an instance, waiting for SSH
    readiness, and unconditional cleanup on exit.
    """

    def __init__(self, api_key: str, arch: str = DEFAULT_ARCHITECTURE) -> None:
        self.sdk = VastAI(api_key, retry=1)
        self.arch = arch
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        self.instance_label = f"{INSTANCE_LABEL_PREFIX}{timestamp}-{uuid4().hex}"
        self.instance_id: int | None = None
        self.ssh_host: str | None = None
        self.ssh_port: int | None = None
        self._atexit_registered = False
        self._creation_ambiguous = False
        self._ssh_key_dir: tempfile.TemporaryDirectory[str] | None = None
        self.ssh_private_key: Path | None = None

    def __enter__(self) -> Self:
        self._log_existing_instances()
        logger.info("This GPU test session owns label=%s", self.instance_label)
        atexit.register(self._atexit_cleanup)
        self._atexit_registered = True
        try:
            self._prepare_ssh_key()
            self._launch()
            self._wait_for_ssh()
        except BaseException:
            try:
                if self._destroy():
                    self._unregister_atexit()
            finally:
                self._cleanup_ssh_key()
            raise
        else:
            return self

    def __exit__(self, *_: object) -> None:
        try:
            if self._destroy():
                self._unregister_atexit()
        finally:
            self._cleanup_ssh_key()

    def _prepare_ssh_key(self) -> None:
        """Generate an ephemeral SSH identity for this test session."""
        self._ssh_key_dir = tempfile.TemporaryDirectory(prefix="warpforth-ssh-")
        self.ssh_private_key = Path(self._ssh_key_dir.name) / "id_ed25519"
        subprocess.run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(self.ssh_private_key),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    def _attach_ssh_key(self) -> None:
        """Attach this session's ephemeral public key to its Vast instance."""
        if self.instance_id is None or self.ssh_private_key is None:
            msg = "Cannot attach SSH key before instance and identity creation"
            raise RuntimeError(msg)

        public_key = self.ssh_private_key.with_suffix(".pub").read_text().strip()
        result = self.sdk.attach_ssh(instance_id=self.instance_id, ssh_key=public_key)
        if not isinstance(result, dict) or result.get("success") is False:
            msg = f"Failed to attach ephemeral SSH key to instance {self.instance_id}"
            raise RuntimeError(msg)
        logger.info("Attached ephemeral SSH key to instance %s", self.instance_id)

    def _cleanup_ssh_key(self) -> None:
        """Remove this session's ephemeral local SSH key material."""
        if self._ssh_key_dir is not None:
            self._ssh_key_dir.cleanup()
            self._ssh_key_dir = None
            self.ssh_private_key = None

    def _show_instances(self) -> list[dict] | None:
        """Return the current account instances, or None if listing fails."""
        try:
            instances = self.sdk.show_instances()
        except Exception:
            logger.exception("Failed to list Vast.ai instances")
            return None

        if not isinstance(instances, list):
            logger.warning("Vast.ai returned an invalid instance listing")
            return None
        return [inst for inst in instances if isinstance(inst, dict)]

    def _log_existing_instances(self) -> None:
        """Log existing WarpForth instances without modifying them."""
        instances = self._show_instances()
        if instances is None:
            logger.warning("Could not check for existing WarpForth Vast.ai instances")
            return

        matching = [
            inst
            for inst in instances
            if str(inst.get("label", "")).startswith(INSTANCE_LABEL_PREFIX)
        ]
        if not matching:
            logger.info("No existing WarpForth Vast.ai instances found")
            return

        total_cost = 0.0
        for inst in matching:
            cost = float(inst.get("dph_total") or 0.0)
            total_cost += cost
            start_date = inst.get("start_date")
            age = "unknown"
            if isinstance(start_date, int | float):
                age = f"{max(0.0, time.time() - start_date) / 60:.0f}m"
            logger.warning(
                "Existing WarpForth Vast.ai instance: id=%s label=%s status=%s "
                "gpu=%s dph=$%.3f age=%s ownership=active run or orphan candidate",
                inst.get("id"),
                inst.get("label"),
                inst.get("actual_status", inst.get("status", "unknown")),
                inst.get("gpu_name", "unknown"),
                cost,
                age,
            )
        logger.warning(
            "Found %d existing WarpForth Vast.ai instance(s), total known cost=$%.3f/hr; "
            "they may belong to other active test runs and will not be destroyed",
            len(matching),
            total_cost,
        )

    def _launch(self) -> None:
        """Find the cheapest suitable offer and launch an instance."""
        minimum_compute_capability = int(self.arch.removeprefix("sm_")) * 10
        query = (
            f"num_gpus=1 rentable=True rented=False compute_cap>={minimum_compute_capability}"
            f" reliability>=0.95 inet_up>=100 dph<={MAX_COST_PER_HOUR}"
            f" cuda_max_good>={CUDA_VERSION}"
        )
        offers = self.sdk.search_offers(query=query, order="dph", limit=5)

        if not offers:
            msg = "No suitable GPU offers found on Vast.ai"
            raise RuntimeError(msg)

        offer = offers[0] if isinstance(offers, list) else offers
        details = offer if isinstance(offer, dict) else {"id": offer}
        offer_id = int(details["id"])
        logger.info(
            "Launching instance from offer %s: gpu=%s driver=%s cuda_max_good=%s",
            offer_id,
            details.get("gpu_name"),
            details.get("driver_version"),
            details.get("cuda_max_good"),
        )
        self._creation_ambiguous = True
        try:
            result = self.sdk.create_instance(
                id=offer_id,
                image=CUDA_IMAGE,
                disk=10.0,
                runtype="ssh_direc ssh_proxy",
                label=self.instance_label,
            )
        except RequestException:
            logger.warning(
                "Instance creation result is ambiguous; reconciling label=%s",
                self.instance_label,
            )
            self._reconcile_instance()
            return

        if (
            isinstance(result, dict)
            and result.get("success") is True
            and result.get("new_contract")
        ):
            self.instance_id = int(result["new_contract"])
            self._creation_ambiguous = False
            logger.info(
                "This GPU test session owns instance id=%s label=%s",
                self.instance_id,
                self.instance_label,
            )
            return

        logger.warning(
            "Instance creation response did not contain an ID; reconciling label=%s",
            self.instance_label,
        )
        self._reconcile_instance()

    def _reconcile_instance(self) -> None:
        """Find this run's instance by its exact unique label."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            instances = self._show_instances()
            if instances is not None:
                matches = [inst for inst in instances if inst.get("label") == self.instance_label]
                if len(matches) > 1:
                    ids = [inst.get("id") for inst in matches]
                    msg = f"Multiple instances found for unique label {self.instance_label}: {ids}"
                    raise RuntimeError(msg)
                if matches:
                    self.instance_id = int(matches[0]["id"])
                    self._creation_ambiguous = False
                    logger.info(
                        "This GPU test session owns instance id=%s label=%s",
                        self.instance_id,
                        self.instance_label,
                    )
                    return
            time.sleep(POLL_INTERVAL_S)

        msg = f"Could not reconcile instance with label {self.instance_label} after creation"
        raise TimeoutError(msg)

    def _wait_for_ssh(self) -> None:
        """Poll until the instance is running and SSH is available."""
        deadline = time.monotonic() + POLL_TIMEOUT_S

        while time.monotonic() < deadline:
            instances = self.sdk.show_instances()
            if not isinstance(instances, list):
                time.sleep(POLL_INTERVAL_S)
                continue

            for inst in instances:
                if not isinstance(inst, dict):
                    continue
                if int(inst.get("id", -1)) != self.instance_id:
                    continue

                status = inst.get("actual_status", inst.get("status", ""))
                if status == "running" and inst.get("ssh_host") and inst.get("ssh_port"):
                    self.ssh_host = inst["ssh_host"]
                    self.ssh_port = int(inst["ssh_port"])
                    logger.info(
                        "Instance %s ready: %s:%s",
                        self.instance_id,
                        self.ssh_host,
                        self.ssh_port,
                    )
                    self._attach_ssh_key()
                    self._wait_for_sshd()
                    self._ensure_python()
                    return

            time.sleep(POLL_INTERVAL_S)

        msg = f"Instance {self.instance_id} did not become SSH-ready within {POLL_TIMEOUT_S}s"
        raise TimeoutError(msg)

    def _wait_for_sshd(self) -> None:
        """Poll until sshd is actually accepting connections."""
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            result = subprocess.run(
                [*self.ssh_argv(), "true"],
                capture_output=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                logger.info("sshd ready on %s:%s", self.ssh_host, self.ssh_port)
                return
            time.sleep(POLL_INTERVAL_S)
        msg = (
            f"sshd on {self.ssh_host}:{self.ssh_port} did not become ready within {POLL_TIMEOUT_S}s"
        )
        raise TimeoutError(msg)

    def _ensure_python(self) -> None:
        """Install python3 for the device worker if the image lacks it."""
        self.ssh_run(
            "command -v python3 >/dev/null || (apt-get update -qq && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3)",
            timeout=600,
        )

    def _record_labeled_instance_ids(
        self,
        instances: list[dict],
        instance_ids: set[int],
    ) -> None:
        """Record instances owned by this session's exact unique label."""
        labeled_ids = {
            int(inst["id"])
            for inst in instances
            if inst.get("label") == self.instance_label and inst.get("id") is not None
        }
        instance_ids.update(labeled_ids)
        if labeled_ids:
            self._creation_ambiguous = False

    def _destroy(self) -> bool:
        """Destroy only this session's instances and verify their absence."""
        instance_ids = {self.instance_id} if self.instance_id is not None else set()
        retries = 3
        for attempt in range(retries):
            instances = self._show_instances()
            if instances is not None:
                self._record_labeled_instance_ids(instances, instance_ids)
                if not self._creation_ambiguous and not instance_ids:
                    logger.info(
                        "Verified cleanup for Vast.ai label=%s instance_ids=%s",
                        self.instance_label,
                        sorted(instance_ids),
                    )
                    self.instance_id = None
                    return True

            for instance_id in instance_ids:
                try:
                    self.sdk.destroy_instance(id=instance_id)
                except Exception:
                    logger.exception(
                        "Destroy attempt %d/%d failed for instance %s",
                        attempt + 1,
                        retries,
                        instance_id,
                    )

            instances = self._show_instances()
            if instances is not None:
                self._record_labeled_instance_ids(instances, instance_ids)
                listed_ids = {int(inst["id"]) for inst in instances if inst.get("id") is not None}
                if not self._creation_ambiguous and not instance_ids & listed_ids:
                    logger.info(
                        "Verified cleanup for Vast.ai label=%s instance_ids=%s",
                        self.instance_label,
                        sorted(instance_ids),
                    )
                    self.instance_id = None
                    return True

            if attempt < retries - 1:
                time.sleep(5)

        logger.error(
            "Could not verify cleanup for Vast.ai label=%s instance_ids=%s; "
            "check the Vast.ai account and destroy these instances manually",
            self.instance_label,
            sorted(instance_ids),
        )
        return False

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_cleanup)
            self._atexit_registered = False

    def _atexit_cleanup(self) -> None:
        """Last-resort cleanup registered before instance creation."""
        logger.warning(
            "atexit: checking for Vast.ai instances with label=%s instance_id=%s",
            self.instance_label,
            self.instance_id,
        )
        try:
            self._destroy()
        finally:
            self._cleanup_ssh_key()

    def _ssh_options(self) -> list[str]:
        if self.ssh_private_key is None:
            msg = "Ephemeral SSH identity is not available"
            raise RuntimeError(msg)
        return [
            "-i",
            str(self.ssh_private_key),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "PreferredAuthentications=publickey",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ConnectionAttempts=3",
            "-o",
            "LogLevel=ERROR",
        ]

    def ssh_argv(self) -> list[str]:
        """Return the ssh argv for this instance; append a remote command to run it."""
        return [
            "ssh",
            *self._ssh_options(),
            "-p",
            str(self.ssh_port),
            f"root@{self.ssh_host}",
        ]

    def ssh_run(self, cmd: str, *, timeout: int = 120) -> str:
        """Execute a command on the remote instance via SSH."""
        result = subprocess.run(
            [*self.ssh_argv(), cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            msg = f"SSH command failed (rc={result.returncode}):\n{result.stderr}"
            raise RuntimeError(msg)
        return result.stdout
