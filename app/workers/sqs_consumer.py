"""SQS consumer loop. In dev (QUEUE_MODE=memory) replaced by in-process queue."""

from __future__ import annotations

from typing import Callable

from .jobs import StepJob


class SQSConsumer:
    def __init__(self, queue_url: str, region: str) -> None:
        self.queue_url = queue_url
        self.region = region

    def run_forever(self, handler: Callable[[StepJob], None]) -> None:
        raise NotImplementedError("Wire boto3 SQS receive_message loop")


class InMemoryQueue:
    def __init__(self) -> None:
        self._q: list[StepJob] = []

    def send(self, job: StepJob) -> None:
        self._q.append(job)

    def drain(self, handler: Callable[[StepJob], None]) -> None:
        while self._q:
            handler(self._q.pop(0))
