"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/deployment/environment.py
Phase    : Phase 8
Status   : IMPLEMENTED

Unified Edge-Cloud deployment environment.
Coordinates EdgeRuntime, CloudRuntime, and NetworkSimulator across Edge, Cloud,
and Two-Level Hybrid execution paths.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from src.decision.base import DecisionAction, DecisionResult, ExecutionResult, ExecutionStatus
from src.deployment.base import DeploymentExecutionResult, TransmissionStatus
from src.deployment.network import NetworkSimulator
from src.deployment.runtimes import CloudRuntime, EdgeRuntime


class DeploymentEnvironment:
    """Integrated deployment environment managing Edge, Cloud, and network execution.

    Coordinates Level 2 execution under the Phase 5 decision policy.
    Maintains fine-grained latency accounting (T_edge, T_cloud, T_network, T_hybrid)
    and explicit failure handling without fabricating predictions.
    """

    def __init__(
        self,
        edge_runtime: EdgeRuntime,
        cloud_runtime: CloudRuntime,
        network_simulator: NetworkSimulator | None = None,
        config: Mapping[str, Any] | None = None,
        fallback_confidence_threshold: float = 0.60,
    ) -> None:
        self.edge_runtime = edge_runtime
        self.cloud_runtime = cloud_runtime
        self.config = dict(config or {})

        if network_simulator is not None:
            self.network_simulator = network_simulator
        else:
            self.network_simulator = NetworkSimulator(config=self.config)

        # Fallback confidence threshold from decision/hybrid config if available
        dec_cfg = self.config.get("decision", {})
        hyb_cfg = dec_cfg.get("hybrid", {})
        self.fallback_confidence_threshold = float(
            hyb_cfg.get("fallback_confidence_threshold", fallback_confidence_threshold)
        )

        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0

    def execute_edge(
        self,
        x: Any,
        observation_index: int | None = None,
        decision: DecisionResult | None = None,
    ) -> DeploymentExecutionResult:
        """Execute inference on the EdgeRuntime only (no network communication)."""
        self._total_executions += 1
        pred, probs, lat_edge, succ, err = self.edge_runtime.execute(x, observation_index)

        if succ:
            self._successful_executions += 1
            status = ExecutionStatus.SUCCESS
        else:
            self._failed_executions += 1
            status = ExecutionStatus.FAILED

        return DeploymentExecutionResult(
            action=DecisionAction.EDGE,
            prediction=pred,
            probabilities=probs,
            model_used="edge",
            success=succ,
            status=status,
            edge_latency_s=lat_edge,
            cloud_latency_s=None,
            network_latency_s=None,
            hybrid_latency_s=None,
            total_latency_s=lat_edge,
            cloud_fallback=False,
            packet_lost=False,
            error=err,
            decision=decision,
            observation_index=observation_index,
        )

    def execute_cloud(
        self,
        x: Any,
        observation_index: int | None = None,
        decision: DecisionResult | None = None,
    ) -> DeploymentExecutionResult:
        """Execute inference on CloudRuntime via NetworkSimulator transmission."""
        self._total_executions += 1

        # 1. Transmit payload across simulated network to Cloud
        tx_res = self.network_simulator.transmit(
            payload=x,
            observation_index=observation_index,
            source="edge",
            destination="cloud",
        )

        if not tx_res.success:
            self._failed_executions += 1
            return DeploymentExecutionResult(
                action=DecisionAction.CLOUD,
                prediction=None,
                probabilities=None,
                model_used="cloud",
                success=False,
                status=ExecutionStatus.FAILED,
                edge_latency_s=None,
                cloud_latency_s=None,
                network_latency_s=tx_res.latency_s,
                hybrid_latency_s=None,
                total_latency_s=tx_res.latency_s,
                cloud_fallback=False,
                packet_lost=tx_res.packet_lost,
                error=tx_res.error,
                decision=decision,
                observation_index=observation_index,
            )

        # 2. Execute Cloud model
        pred, probs, lat_cloud, succ, err = self.cloud_runtime.execute(x, observation_index)
        total_lat = tx_res.latency_s + lat_cloud

        if succ:
            self._successful_executions += 1
            status = ExecutionStatus.SUCCESS
        else:
            self._failed_executions += 1
            status = ExecutionStatus.FAILED

        return DeploymentExecutionResult(
            action=DecisionAction.CLOUD,
            prediction=pred,
            probabilities=probs,
            model_used="cloud",
            success=succ,
            status=status,
            edge_latency_s=None,
            cloud_latency_s=lat_cloud,
            network_latency_s=tx_res.latency_s,
            hybrid_latency_s=None,
            total_latency_s=total_lat,
            cloud_fallback=False,
            packet_lost=False,
            error=err,
            decision=decision,
            observation_index=observation_index,
        )

    def transmit_to_cloud(
        self,
        x: Any,
        observation_index: int | None = None,
    ) -> Any:
        """Expose direct network transmission capability."""
        return self.network_simulator.transmit(x, observation_index=observation_index)

    def execute_hybrid(
        self,
        x: Any,
        observation_index: int | None = None,
        decision: DecisionResult | None = None,
        fallback_threshold: float | None = None,
    ) -> DeploymentExecutionResult:
        """Execute Two-Level Hybrid path: Edge first -> evaluate confidence -> conditional Cloud fallback."""
        self._total_executions += 1
        threshold = fallback_threshold if fallback_threshold is not None else self.fallback_confidence_threshold

        t_start_hybrid = time.perf_counter()

        # Step 1: Execute Edge model
        pred_e, probs_e, lat_e, succ_e, err_e = self.edge_runtime.execute(x, observation_index)

        if not succ_e:
            t_hyb_total = time.perf_counter() - t_start_hybrid
            self._failed_executions += 1
            return DeploymentExecutionResult(
                action=DecisionAction.HYBRID,
                prediction=None,
                probabilities=None,
                model_used="hybrid_edge",
                success=False,
                status=ExecutionStatus.FAILED,
                edge_latency_s=lat_e,
                cloud_latency_s=None,
                network_latency_s=None,
                hybrid_latency_s=t_hyb_total,
                total_latency_s=t_hyb_total,
                cloud_fallback=False,
                packet_lost=False,
                error=f"Hybrid Edge execution failure: {err_e}",
                decision=decision,
                observation_index=observation_index,
            )

        # Step 2: Calculate Edge confidence C_edge = 2 * (max(P0, P1) - 0.5)
        p0 = probs_e.get(0, 0.5) if probs_e else 0.5
        p1 = probs_e.get(1, 0.5) if probs_e else 0.5
        c_edge = 2.0 * (max(p0, p1) - 0.5)

        if c_edge >= threshold:
            # Confident Edge inference: complete at Edge without Cloud invocation or network transmission
            t_hyb_total = time.perf_counter() - t_start_hybrid
            self._successful_executions += 1
            return DeploymentExecutionResult(
                action=DecisionAction.HYBRID,
                prediction=pred_e,
                probabilities=probs_e,
                model_used="hybrid_edge",
                success=True,
                status=ExecutionStatus.SUCCESS,
                edge_latency_s=lat_e,
                cloud_latency_s=None,
                network_latency_s=None,
                hybrid_latency_s=t_hyb_total,
                total_latency_s=t_hyb_total,
                cloud_fallback=False,
                packet_lost=False,
                error=None,
                decision=decision,
                observation_index=observation_index,
            )

        # Step 3: Edge uncertain (C_edge < threshold) -> Fallback to Cloud across network
        tx_res = self.network_simulator.transmit(
            payload=x,
            observation_index=observation_index,
            source="edge",
            destination="cloud",
        )

        if not tx_res.success:
            t_hyb_total = time.perf_counter() - t_start_hybrid
            self._failed_executions += 1
            return DeploymentExecutionResult(
                action=DecisionAction.HYBRID,
                prediction=None,
                probabilities=None,
                model_used="hybrid_cloud",
                success=False,
                status=ExecutionStatus.FAILED,
                edge_latency_s=lat_e,
                cloud_latency_s=None,
                network_latency_s=tx_res.latency_s,
                hybrid_latency_s=t_hyb_total,
                total_latency_s=t_hyb_total,
                cloud_fallback=True,
                packet_lost=tx_res.packet_lost,
                error=f"Hybrid Cloud fallback network failure: {tx_res.error}",
                decision=decision,
                observation_index=observation_index,
            )

        # Step 4: Execute Cloud model after successful transmission
        pred_c, probs_c, lat_c, succ_c, err_c = self.cloud_runtime.execute(x, observation_index)
        t_hyb_total = time.perf_counter() - t_start_hybrid

        if not succ_c:
            self._failed_executions += 1
            return DeploymentExecutionResult(
                action=DecisionAction.HYBRID,
                prediction=None,
                probabilities=None,
                model_used="hybrid_cloud",
                success=False,
                status=ExecutionStatus.FAILED,
                edge_latency_s=lat_e,
                cloud_latency_s=lat_c,
                network_latency_s=tx_res.latency_s,
                hybrid_latency_s=t_hyb_total,
                total_latency_s=t_hyb_total,
                cloud_fallback=True,
                packet_lost=False,
                error=f"Hybrid Cloud model execution failure: {err_c}",
                decision=decision,
                observation_index=observation_index,
            )

        self._successful_executions += 1
        return DeploymentExecutionResult(
            action=DecisionAction.HYBRID,
            prediction=pred_c,
            probabilities=probs_c,
            model_used="hybrid_cloud",
            success=True,
            status=ExecutionStatus.FALLBACK,
            edge_latency_s=lat_e,
            cloud_latency_s=lat_c,
            network_latency_s=tx_res.latency_s,
            hybrid_latency_s=t_hyb_total,
            total_latency_s=t_hyb_total,
            cloud_fallback=True,
            packet_lost=False,
            error=None,
            decision=decision,
            observation_index=observation_index,
        )

    def execute(
        self,
        action: DecisionAction,
        x: Any,
        decision: DecisionResult | None = None,
        observation_index: int | None = None,
    ) -> ExecutionResult:
        """Unified dispatching interface converting outcome to standard Phase 6/7 ExecutionResult."""
        act = action if isinstance(action, DecisionAction) else DecisionAction.from_str(action)

        if act == DecisionAction.EDGE:
            dep_res = self.execute_edge(x, observation_index=observation_index, decision=decision)
        elif act == DecisionAction.CLOUD:
            dep_res = self.execute_cloud(x, observation_index=observation_index, decision=decision)
        elif act == DecisionAction.HYBRID:
            dep_res = self.execute_hybrid(x, observation_index=observation_index, decision=decision)
        else:
            raise ValueError(f"Unknown action: {action}")

        return dep_res.to_execution_result()

    def get_stats(self) -> dict[str, Any]:
        """Return aggregated execution and environment statistics."""
        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "failed_executions": self._failed_executions,
            "edge": self.edge_runtime.get_stats(),
            "cloud": self.cloud_runtime.get_stats(),
            "network": self.network_simulator.get_stats(),
        }

    def reset(self) -> None:
        """Reset execution counters and child runtime/network states."""
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
        self.edge_runtime.reset()
        self.cloud_runtime.reset()
        self.network_simulator.reset()
