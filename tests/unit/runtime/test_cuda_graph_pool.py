import torch

from nestquant.runtime.cuda_graph_decode import NQRTGraphDecodePool


class FakeRunner:
    def __init__(self, model, *, batch_size: int, max_cache_len: int) -> None:
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.prefill_input = None
        self.capture_input = None
        self.decode_input = None

    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.prefill_input = input_ids.clone()
        return input_ids[:, -1:].to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)

    def capture(self, token: torch.Tensor, *, position=None) -> None:
        self.capture_input = (token.clone(), position)

    def decode(self, token: torch.Tensor, *, position: int) -> torch.Tensor:
        self.decode_input = (token.clone(), position)
        return token.to(torch.float32).unsqueeze(-1).expand(-1, -1, 3)

    def decode_eager(self, token: torch.Tensor, *, position: int) -> torch.Tensor:
        return self.decode(token, position=position)


def make_pool() -> NQRTGraphDecodePool:
    return NQRTGraphDecodePool(
        torch.nn.Linear(1, 1), batch_sizes=(1, 2, 4, 8), max_cache_len=32,
        runner_factory=FakeRunner,
    )


def test_capacity_selects_smallest_fitting_graph() -> None:
    pool = make_pool()
    assert [pool.select_capacity(value) for value in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]


def test_prefill_pads_and_slices_inactive_rows() -> None:
    pool = make_pool()
    inputs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    logits = pool.prefill(inputs, pad_token_id=99)
    assert pool.active_count == 3
    assert pool.capacity == 4
    assert logits.shape == (3, 1, 3)
    assert pool.runners[4].prefill_input.tolist() == [[1, 2], [3, 4], [5, 6], [99, 99]]


def test_capture_and_decode_use_same_capacity_and_hide_padding() -> None:
    pool = make_pool()
    pool.prefill(torch.tensor([[1, 2], [3, 4], [5, 6]]))
    token = torch.tensor([[7], [8], [9]])
    pool.capture(token, position=2, pad_token_id=42)
    logits = pool.decode(token, position=2, pad_token_id=42)
    eager_logits = pool.decode_eager(token, position=2, pad_token_id=42)
    assert logits.shape == (3, 1, 3)
    assert torch.equal(logits, eager_logits)
    assert pool.runners[4].capture_input[0].flatten().tolist() == [7, 8, 9, 42]
    assert pool.runners[4].decode_input[0].flatten().tolist() == [7, 8, 9, 42]


def test_pool_rejects_invalid_capacity_and_batch_mismatch() -> None:
    pool = make_pool()
    for invalid in (0, 9):
        try:
            pool.select_capacity(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid active_count was accepted")
    pool.prefill(torch.tensor([[1], [2], [3]]))
    try:
        pool.decode(torch.tensor([[1], [2]]), position=1)
    except ValueError:
        pass
    else:
        raise AssertionError("batch mismatch was accepted")
