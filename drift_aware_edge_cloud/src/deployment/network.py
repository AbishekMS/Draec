"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/deployment/network.py
Phase    : Phase 8
Status   : IMPLEMENTED

Configurable network simulation and emulation component.
Models communication delay, jitter, packet loss, and link availability deterministically.
"""

from __future__ import annotations

import random
import time
from typing import Any, Mapping, Sequence

from src.deployment.base import NetworkPacket, TransmissionResult, TransmissionStatus


class NetworkSimulator:
    """Configurable software network simulator for Edge-to-Cloud communication.

    Simulates network transmission delay, jitter, and packet loss.
    Does NOT claim to represent physical hardware measurements.
    Strictly avoids physical blocking (time.sleep) unless pacing is explicitly enabled.
    """

    def __init__(
        self,
        base_latency_s: float = 0.020,
        jitter_s: float = 0.005,
        packet_loss_probability: float = 0.0,
        available: bool = True,
        seed: int | None = 42,
        pacing_enabled: bool = False,
        failure_schedule: Sequence[int] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if config is not None:
            net_cfg = config.get("network", {})
            lat_cfg = net_cfg.get("latency", {})
            base_latency_s = float(lat_cfg.get("base_s", base_latency_s))
            jitter_s = float(lat_cfg.get("jitter_s", jitter_s))
            loss_cfg = net_cfg.get("packet_loss", {})
            packet_loss_probability = float(loss_cfg.get("probability", packet_loss_probability))
            avail_cfg = net_cfg.get("availability", {})
            available = bool(avail_cfg.get("network", available))
            det_cfg = net_cfg.get("deterministic", {})
            if det_cfg.get("enabled", True):
                seed = det_cfg.get("seed", seed)
            pacing_enabled = bool(net_cfg.get("pacing_enabled", pacing_enabled))

        self.base_latency_s = max(0.0, float(base_latency_s))
        self.jitter_s = max(0.0, float(jitter_s))
        self.packet_loss_probability = max(0.0, min(1.0, float(packet_loss_probability)))
        self.available = bool(available)
        self.seed = int(seed) if seed is not None else None
        self.pacing_enabled = bool(pacing_enabled)
        self.failure_schedule = set(failure_schedule) if failure_schedule is not None else set()

        self._rng = random.Random(self.seed)
        self._seq_counter = 0
        self._total_transmissions = 0
        self._successful_transmissions = 0
        self._lost_transmissions = 0
        self._disconnected_transmissions = 0

    def transmit(
        self,
        payload: Any,
        observation_index: int | None = None,
        source: str = "edge",
        destination: str = "cloud",
    ) -> TransmissionResult:
        """Simulate transmitting a payload between source and destination."""
        self._total_transmissions += 1
        self._seq_counter += 1

        # 1. Check link availability
        if not self.available:
            self._disconnected_transmissions += 1
            return TransmissionResult(
                status=TransmissionStatus.DISCONNECTED,
                success=False,
                latency_s=0.0,
                packet_lost=False,
                error="Network transmission failed: link unavailable / disconnected",
            )

        # 2. Check deterministic failure schedule
        if observation_index is not None and observation_index in self.failure_schedule:
            self._lost_transmissions += 1
            return TransmissionResult(
                status=TransmissionStatus.PACKET_LOSS,
                success=False,
                latency_s=0.0,
                packet_lost=True,
                error="Network transmission failed: scheduled packet loss",
            )

        # 3. Evaluate pseudo-random packet loss
        if self.packet_loss_probability > 0.0:
            draw = self._rng.random()
            if draw < self.packet_loss_probability:
                self._lost_transmissions += 1
                return TransmissionResult(
                    status=TransmissionStatus.PACKET_LOSS,
                    success=False,
                    latency_s=0.0,
                    packet_lost=True,
                    error=f"Network transmission failed: simulated packet loss (p={self.packet_loss_probability:.3f})",
                )

        # 4. Compute simulated network transmission latency: T_network = T_base + jitter
        if self.jitter_s > 0.0:
            jitter = self._rng.uniform(-self.jitter_s, self.jitter_s)
        else:
            jitter = 0.0
        sim_latency = max(0.0, self.base_latency_s + jitter)

        # Optional real-time physical pacing (disabled by default)
        if self.pacing_enabled and sim_latency > 0.0:
            time.sleep(sim_latency)

        self._successful_transmissions += 1
        approx_bytes = getattr(payload, "nbytes", getattr(payload, "__sizeof__", lambda: 128)())

        return TransmissionResult(
            status=TransmissionStatus.DELIVERED,
            success=True,
            latency_s=sim_latency,
            packet_lost=False,
            bytes_transferred=int(approx_bytes),
        )

    def set_availability(self, available: bool) -> None:
        """Toggle network link availability state."""
        self.available = bool(available)

    def set_packet_loss_probability(self, prob: float) -> None:
        """Update the packet loss probability dynamically."""
        self.packet_loss_probability = max(0.0, min(1.0, float(prob)))

    def schedule_failure(self, observation_index: int) -> None:
        """Add an observation index to the deterministic failure schedule."""
        self.failure_schedule.add(int(observation_index))

    def clear_schedule(self) -> None:
        """Clear all scheduled failures."""
        self.failure_schedule.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return cumulative network transmission statistics."""
        loss_rate = (
            self._lost_transmissions / self._total_transmissions
            if self._total_transmissions > 0
            else 0.0
        )
        return {
            "total_transmissions": self._total_transmissions,
            "successful_transmissions": self._successful_transmissions,
            "lost_transmissions": self._lost_transmissions,
            "disconnected_transmissions": self._disconnected_transmissions,
            "loss_rate": loss_rate,
            "available": self.available,
            "base_latency_s": self.base_latency_s,
            "jitter_s": self.jitter_s,
            "packet_loss_probability": self.packet_loss_probability,
            "pacing_enabled": self.pacing_enabled,
        }

    def reset(self) -> None:
        """Reset sequence counters and PRNG state."""
        self._rng = random.Random(self.seed)
        self._seq_counter = 0
        self._total_transmissions = 0
        self._successful_transmissions = 0
        self._lost_transmissions = 0
        self._disconnected_transmissions = 0
        self.failure_schedule.clear()
