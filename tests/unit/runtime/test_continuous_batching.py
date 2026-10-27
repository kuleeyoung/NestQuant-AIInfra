import pytest

from nestquant.runtime.continuous_batching import (
    ContinuousBatchScheduler,
    DecodeRequest,
    RequestStatus,
)


def request(name: str, *, limit: int = 3, eos: int | None = None) -> DecodeRequest:
    return DecodeRequest(name, (1, 2), limit, eos_token_id=eos)


def test_scheduler_admits_waiting_requests_and_reuses_finished_slot() -> None:
    scheduler = ContinuousBatchScheduler(2)
    first, second, third = request("a", limit=1), request("b"), request("c")
    for item in (first, second, third):
        scheduler.submit(item)
    assert scheduler.admit() == (first, second)
    assert scheduler.pending_count() == 1
    assert scheduler.record_tokens({0: 9}) == (first,)
    assert scheduler.consume_released_slots() == (0,)
    assert scheduler.admit() == (third,)
    assert third.slot == 0
    assert scheduler.active_mask() == (True, True)


def test_scheduler_tracks_independent_positions() -> None:
    scheduler = ContinuousBatchScheduler(2)
    first = DecodeRequest("a", (1, 2, 3), 4)
    second = DecodeRequest("b", (4,), 4)
    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.admit()
    scheduler.record_tokens({0: 8})
    assert scheduler.positions() == (3, 1)


def test_eos_and_cancel_are_terminal_and_release_slots() -> None:
    scheduler = ContinuousBatchScheduler(2)
    first, second = request("a", eos=7), request("b")
    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.admit()
    scheduler.record_tokens({0: 7})
    scheduler.cancel("b")
    assert first.status is RequestStatus.FINISHED
    assert second.status is RequestStatus.CANCELLED
    assert scheduler.consume_released_slots() == (0, 1)
    assert scheduler.running == ()


def test_duplicate_id_and_unknown_slot_fail_loudly() -> None:
    scheduler = ContinuousBatchScheduler(1)
    scheduler.submit(request("a"))
    with pytest.raises(ValueError, match="duplicate"):
        scheduler.submit(request("a"))
    with pytest.raises(ValueError, match="not running"):
        scheduler.record_tokens({0: 1})


def test_terminal_request_can_be_forgotten_but_running_request_cannot() -> None:
    scheduler = ContinuousBatchScheduler(1)
    item = request("a", limit=1)
    scheduler.submit(item)
    scheduler.admit()
    with pytest.raises(ValueError, match="terminal"):
        scheduler.forget_terminal("a")
    scheduler.record_tokens({0: 9})
    assert scheduler.forget_terminal("a") is item
    assert scheduler.registered_count() == 0
