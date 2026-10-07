import torch

from src.models import BiLSTMMultiTask, TransformerMultiTask
from src.training.trainer import multitask_loss
from tests.conftest import INTENTS, TINY_ENC

LAMBDAS = {"intent": 1.0, "urgency": 1.0, "escalation": 1.0}


def _batch(vocab=60, b=4, t=7):
    ids = torch.randint(5, vocab, (b, t))
    mask = torch.ones(b, t, dtype=torch.long)
    mask[0, 4:] = 0
    ids[0, 4:] = 0
    return ids, mask


def test_bilstm_output_shapes():
    m = BiLSTMMultiTask(60, len(INTENTS), emb_dim=8, hidden=8, shared_dim=16)
    out = m(*_batch())
    assert out["intent"].shape == (4, len(INTENTS)) and out["urgency"].shape == (4, 5) and out["escalation"].shape == (4,)


def test_transformer_output_shapes():
    m = TransformerMultiTask("tiny", len(INTENTS), pretrained=False, encoder_config=dict(TINY_ENC))
    out = m(*_batch())
    assert out["intent"].shape == (4, len(INTENTS)) and out["urgency"].shape == (4, 5) and out["escalation"].shape == (4,)


def test_padding_does_not_change_bilstm_output():
    torch.manual_seed(0)
    m = BiLSTMMultiTask(60, 5, emb_dim=8, hidden=8, shared_dim=16).eval()
    ids = torch.tensor([[5, 6, 7, 0, 0]]); mask = torch.tensor([[1, 1, 1, 0, 0]])
    a = m(ids, mask)["intent"]
    b = m(ids[:, :3], mask[:, :3])["intent"]
    assert torch.allclose(a, b, atol=1e-5)


def test_multitask_loss_masks_missing_labels_and_backprops():
    m = BiLSTMMultiTask(60, 5, emb_dim=8, hidden=8, shared_dim=16)
    ids, mask = _batch()
    batch = {"intent": torch.tensor([1, -1, 2, -1]), "urgency": torch.tensor([-1, 3, -1, -1]),
             "escalation": torch.tensor([-1, 1, -1, -1])}
    loss, parts = multitask_loss(m(ids, mask), batch, LAMBDAS, "cpu")
    assert torch.isfinite(loss)
    loss.backward()
    assert m.intent_head.weight.grad.abs().sum() > 0 and m.escalation_head.weight.grad.abs().sum() > 0


def test_loss_with_no_labels_for_a_task_is_zero_not_nan():
    m = BiLSTMMultiTask(60, 5, emb_dim=8, hidden=8, shared_dim=16)
    ids, mask = _batch()
    batch = {"intent": torch.tensor([1, 2, 0, 1]), "urgency": torch.full((4,), -1), "escalation": torch.full((4,), -1)}
    loss, parts = multitask_loss(m(ids, mask), batch, LAMBDAS, "cpu")
    assert torch.isfinite(loss) and float(parts["urgency"]) == 0.0 and float(parts["escalation"]) == 0.0
