"""Append-only evidence storage with versioned role views."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from .contracts import EvidenceItem, EvidenceView


class EvidenceLedger:
    """Stores exact evidence items without allowing replacement."""

    def __init__(
        self,
        *,
        record_id: str,
        task: str,
        taxonomy: dict[str, list[str]],
        domain_profile: str,
        initial_items: Iterable[EvidenceItem] = (),
    ) -> None:
        if not record_id:
            raise ValueError("record_id must not be empty")
        if not task:
            raise ValueError("task must not be empty")
        self._record_id = record_id
        self._task = task
        self._taxonomy = {
            dimension: dict(labels) if isinstance(labels, Mapping) else list(labels)
            for dimension, labels in taxonomy.items()
        }
        self._domain_profile = domain_profile
        self._items: dict[str, EvidenceItem] = {}
        self._version = 0
        for item in initial_items:
            self.append(item)

    @property
    def record_id(self) -> str:
        return self._record_id

    @property
    def version(self) -> int:
        return self._version

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def append(self, item: EvidenceItem) -> int:
        if item.record_id != self._record_id:
            raise ValueError(
                f"evidence record_id {item.record_id!r} does not match "
                f"ledger record_id {self._record_id!r}"
            )
        if item.evidence_id in self._items:
            raise ValueError(f"evidence_id {item.evidence_id!r} already exists")
        self._items[item.evidence_id] = item
        self._version += 1
        return self._version

    def get(self, evidence_id: str) -> EvidenceItem:
        try:
            return self._items[evidence_id]
        except KeyError as error:
            raise KeyError(f"unknown evidence_id {evidence_id!r}") from error

    def view(self, evidence_ids: Sequence[str] | None = None) -> EvidenceView:
        selected_ids = self.evidence_ids if evidence_ids is None else evidence_ids
        return EvidenceView(
            record_id=self._record_id,
            task=self._task,
            taxonomy={
                dimension: dict(labels) if isinstance(labels, Mapping) else list(labels)
                for dimension, labels in self._taxonomy.items()
            },
            domain_profile=self._domain_profile,
            ledger_version=self._version,
            items=tuple(self.get(evidence_id) for evidence_id in selected_ids),
        )
