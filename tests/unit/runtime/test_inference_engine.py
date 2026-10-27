import torch
import random

from nestquant.runtime.continuous_batching import DecodeRequest, RequestStatus
from nestquant.runtime.inference_engine import AdmissionPolicy, ContinuousBatchInferenceEngine


class FakeBackend:
    def __init__(self, batch_size: int = 2, max_cache_len: int = 16) -> None:
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.device = torch.device("cpu")
        self.clear_calls = []
        self.decode_positions = []
        self.prefill_positions = []

    @staticmethod
    def _logits(input_ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(*input_ids.shape, 32)
        next_ids = (input_ids + 1).clamp_max(31)
        return logits.scatter(-1, next_ids.unsqueeze(-1), 1.0)

    def prefill(self, input_ids):
        return self._logits(input_ids)

    def prefill_slots(self, input_ids, **kwargs):
        self.prefill_positions.append(kwargs["positions"].clone())
        return self._logits(input_ids)

    def capture(self, sample_token, **kwargs):
        return None

    def decode(self, token, *, positions, **kwargs):
        self.decode_positions.append(positions.clone())
        return self._logits(token)

    def clear_slots(self, slot_mask):
        self.clear_calls.append(slot_mask.clone())


def test_engine_generates_one_token_per_request_per_step() -> None:
    engine = ContinuousBatchInferenceEngine(FakeBackend())
    first = DecodeRequest("a", (1, 2), 3)
    second = DecodeRequest("b", (5,), 2)
    engine.submit(first)
    engine.submit(second)
    step_1 = engine.step()
    assert step_1.generated_tokens == {"a": 3, "b": 6}
    assert first.generated_token_ids == [3]
    step_2 = engine.step()
    assert step_2.generated_tokens == {"a": 4, "b": 7}
    assert second.status is RequestStatus.FINISHED
    step_3 = engine.step()
    assert step_3.generated_tokens == {"a": 5}
    assert first.status is RequestStatus.FINISHED
    assert not engine.has_work()


def test_engine_uses_correct_next_write_positions() -> None:
    backend = FakeBackend()
    engine = ContinuousBatchInferenceEngine(backend)
    engine.submit(DecodeRequest("long", (1, 2, 3), 3))
    engine.submit(DecodeRequest("short", (7,), 3))
    engine.step()
    engine.step()
    positions = backend.decode_positions[-1][:, 0].tolist()
    assert positions == [3, 1]


def test_variable_prompt_padding_uses_unique_scatter_positions() -> None:
    backend = FakeBackend()
    engine = ContinuousBatchInferenceEngine(backend)
    engine.submit(DecodeRequest("long", (1, 2, 3, 4), 1))
    engine.submit(DecodeRequest("short", (7, 8), 1))
    engine.step()
    positions = backend.prefill_positions[-1]
    assert positions[0].tolist() == [0, 1, 2, 3]
    assert positions[1].tolist() == [0, 1, 2, 3]


def test_engine_clears_finished_slot_before_reuse() -> None:
    backend = FakeBackend(batch_size=1)
    engine = ContinuousBatchInferenceEngine(backend)
    first = DecodeRequest("first", (1,), 1)
    second = DecodeRequest("second", (9,), 1)
    engine.submit(first)
    engine.submit(second)
    assert engine.step().finished_request_ids == ("first",)
    second_step = engine.step()
    assert second_step.admitted_request_ids == ("second",)
    assert backend.clear_calls[-1].tolist() == [True]


def test_engine_rejects_request_larger_than_cache() -> None:
    engine = ContinuousBatchInferenceEngine(FakeBackend(max_cache_len=4))
    try:
        engine.submit(DecodeRequest("too-long", (1, 2, 3), 2))
    except ValueError as error:
        assert "exceeds" in str(error)
    else:
        raise AssertionError("oversized request was accepted")


def test_randomized_workload_drains_without_cross_request_tokens() -> None:
    rng = random.Random(20261026)
    engine = ContinuousBatchInferenceEngine(FakeBackend(batch_size=4, max_cache_len=32))
    requests = []
    for index in range(200):
        last_prompt_token = rng.randint(1, 20)
        prompt = tuple([rng.randint(1, 20), last_prompt_token])
        limit = rng.randint(1, 6)
        item = DecodeRequest(f"request-{index}", prompt, limit)
        requests.append(item)
        engine.submit(item)
    steps = 0
    while engine.has_work():
        engine.step()
        steps += 1
        assert steps < 1000
    for item in requests:
        expected = [min(item.prompt_token_ids[-1] + offset, 31) for offset in range(1, item.max_new_tokens + 1)]
        assert item.generated_token_ids == expected
        assert item.status is RequestStatus.FINISHED


def test_microbatch_waits_for_threshold_then_admits_together() -> None:
    backend = FakeBackend(batch_size=4)
    engine = ContinuousBatchInferenceEngine(
        backend, admission_policy=AdmissionPolicy(min_prefill_batch=2, max_wait_steps=5)
    )
    running = DecodeRequest("running", (1,), 4)
    engine.submit(running)
    engine.step()
    first_waiter = DecodeRequest("wait-1", (2,), 1)
    engine.submit(first_waiter)
    delayed = engine.step()
    assert delayed.admitted_request_ids == ()
    second_waiter = DecodeRequest("wait-2", (3,), 1)
    engine.submit(second_waiter)
    admitted = engine.step()
    assert admitted.admitted_request_ids == ("wait-1", "wait-2")


def test_microbatch_deadline_and_idle_fast_path() -> None:
    backend = FakeBackend(batch_size=2)
    engine = ContinuousBatchInferenceEngine(
        backend, admission_policy=AdmissionPolicy(min_prefill_batch=2, max_wait_steps=1)
    )
    first = DecodeRequest("first", (1,), 3)
    engine.submit(first)
    assert engine.step().admitted_request_ids == ("first",)
    waiter = DecodeRequest("waiter", (2,), 1)
    engine.submit(waiter)
    assert engine.step().admitted_request_ids == ()
    assert engine.step().admitted_request_ids == ("waiter",)
