import math, torch
from yev.train.losses import ce_soft, full_vocab_ce, pair_margin, perm_skl, rps

def test_ce_soft_ignores_masked_letters_and_weights_rows():
    logits = torch.tensor([[2.0, 0.0, 99.0], [0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    w = torch.tensor([1.0, 0.0])
    got = ce_soft(logits, target, mask, w)
    exp = -math.log(math.exp(2) / (math.exp(2) + 1))
    assert math.isclose(got.item(), exp, rel_tol=1e-5)

def test_full_vocab_ce_matches_cross_entropy():
    v = torch.randn(3, 11); ids = torch.tensor([1, 5, 7]); w = torch.ones(3)
    assert torch.allclose(full_vocab_ce(v, ids, w), torch.nn.functional.cross_entropy(v, ids))

def test_pair_margin_zero_when_separated():
    logp = torch.log(torch.tensor([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]]))
    # row0 gold key X at index0; row1 gold key Y at index1; key X in row1 is index0, key Y in row0 is index1
    assert pair_margin(logp, [(0, 1, 0, 0, 1, 1)], margin=1.0).item() == 0.0

def test_pair_margin_positive_when_identical():
    logp = torch.log(torch.full((2, 2), 0.5))
    assert pair_margin(logp, [(0, 1, 0, 0, 1, 1)], margin=1.0).item() == 2.0

def test_perm_skl_zero_for_equal_and_empty():
    p = torch.tensor([[0.2, 0.8]])
    assert perm_skl(p, p).item() == 0.0
    assert perm_skl(torch.zeros(0, 2), torch.zeros(0, 2)).item() == 0.0

def test_rps_perfect_is_zero_and_far_is_larger():
    mask = torch.ones(1, 3, dtype=torch.bool)
    t = torch.tensor([[0.0, 0.0, 1.0]])
    assert rps(t.clone(), t, mask).item() == 0.0
    near = rps(torch.tensor([[0.0, 1.0, 0.0]]), t, mask).item()
    far = rps(torch.tensor([[1.0, 0.0, 0.0]]), t, mask).item()
    assert far > near > 0
