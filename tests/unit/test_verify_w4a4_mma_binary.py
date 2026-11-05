from scripts.verify_w4a4_mma_binary import find_hmma_lines


def test_finds_only_target_bf16_hmma_instruction():
    sass = """
        /*100*/ HMMA.16816.F32.BF16 R12, R16, R30, R12 ;
        /*110*/ FFMA R1, R2, R3, R4 ;
        /*120*/ HMMA.16816.F32.F16 R12, R16, R30, R12 ;
    """
    matches = find_hmma_lines(sass)
    assert len(matches) == 1
    assert "HMMA.16816.F32.BF16" in matches[0]
