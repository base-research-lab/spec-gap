"""Trajectory logging -- writes one JSONL record stream per trajectory.

Captures every agent decision, tool call, and inter-agent message with
timestamps and call-graph structure. One file per trajectory run.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schema import TrajectoryRecord


class TrajectoryLogger:
    def __init__(self, output_dir: str = "trajectories", trajectory_id: str = None):
        self.trajectory_id = trajectory_id or str(uuid.uuid4())
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.step_index = 0
        self.records: list[TrajectoryRecord] = []
        self._path = self.output_dir / f"{self.trajectory_id}.jsonl"

    def log_step(
        self,
        node_id: str,
        role: str,
        model: str,
        timestamp_start: str,
        timestamp_end: str,
        input_context: list,
        output_message: str,
        inter_agent_msgs: list = None,
        tool_calls: list = None,
        call_graph_edges: list = None,
        injection_point: str = None,
        token_position: str = "last",
        hop_mode: str = "2-hop",
        trust_mode: str = "same-model",
        status: str = "completed",
        scenario_id: str = None,
        condition: str = None,
        injection_wording_id: str = None,
        contrast_pair_id: str = None,
        failure_mode: str = None,
        trajectory_label: str = None,
        step_label: str = None,
        behavioral_sanity: dict = None,
        label_notes: str = None,
    ) -> TrajectoryRecord:
        record = TrajectoryRecord(
            trajectory_id=self.trajectory_id,
            step_index=self.step_index,
            node_id=node_id,
            role=role,
            model=model,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            input_context=input_context,
            output_message=output_message,
            inter_agent_msgs=inter_agent_msgs or [],
            tool_calls=tool_calls or [],
            call_graph_edges=call_graph_edges or [],
            injection_point=injection_point,
            token_position=token_position,
            hop_mode=hop_mode,
            trust_mode=trust_mode,
            status=status,
            scenario_id=scenario_id,
            condition=condition,
            injection_wording_id=injection_wording_id,
            contrast_pair_id=contrast_pair_id,
            failure_mode=failure_mode,
            trajectory_label=trajectory_label,
            step_label=step_label,
            behavioral_sanity=behavioral_sanity,
            label_notes=label_notes,
        )
        self.records.append(record)
        self.step_index += 1

        with open(self._path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    @property
    def path(self) -> Path:
        return self._path

    def validate(self) -> list[str]:
        """Check that the written JSONL conforms to the schema."""
        errors = []
        required_fields = {
            "trajectory_id", "step_index", "node_id", "role", "model",
            "timestamp_start", "timestamp_end", "input_context",
            "output_message", "inter_agent_msgs", "tool_calls",
            "call_graph_edges", "injection_point", "token_position",
            "hop_mode", "trust_mode", "status",
        }
        with open(self._path) as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                missing = required_fields - set(record.keys())
                if missing:
                    errors.append(f"Record {i}: missing fields {missing}")
                if record["step_index"] != i:
                    errors.append(
                        f"Record {i}: step_index={record['step_index']}, "
                        f"expected {i}"
                    )
        return errors


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
