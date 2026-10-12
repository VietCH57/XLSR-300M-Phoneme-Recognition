import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

from phonrec.data import num_output_frames
from phonrec.decoding import ctc_greedy_decode
from phonrec.models import XLSRPhonemeCTC


def tiny_encoder():
    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16,) * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                         feat_extract_norm="layer", do_stable_layer_norm=True, mask_time_prob=0.05)
    return Wav2Vec2Model(cfg)


def test_forward_loss_decode_and_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = XLSRPhonemeCTC(tiny_encoder(), vocab_size=79, blank_id=78)
    model.freeze_feature_encoder()
    x = torch.randn(2, 16000)
    mask = torch.ones(2, 16000, dtype=torch.long)
    mask[1, 8000:] = 0
    labels = torch.tensor([[1, 2, 3], [4, -100, -100]])
    out = model(x, mask, labels=labels)
    assert out["log_probs"].shape[-1] == 79
    assert out["output_lengths"].tolist() == [num_output_frames(16000), num_output_frames(8000)]
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.head.weight.grad is not None
    assert all(p.grad is None for p in model.encoder.feature_extractor.parameters())

    model.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in model.encoder.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.set_backbone_trainable(True)
    assert not any(p.requires_grad for p in model.encoder.feature_extractor.parameters())

    model.eval()
    model.save(tmp_path / "ck")
    re = XLSRPhonemeCTC.from_checkpoint(tmp_path / "ck").eval()
    with torch.no_grad():
        a = model(x, mask)["log_probs"]
        b = re(x, mask)["log_probs"]
    assert torch.allclose(a, b, atol=1e-5)


def test_greedy_decode():
    blank = 3
    seq = torch.tensor([[0, 0, 3, 0, 1, 1, 3, 3, 2, 2]])
    lp = torch.nn.functional.one_hot(seq, 4).float().log()
    assert ctc_greedy_decode(lp, torch.tensor([10]), blank) == [[0, 0, 1, 2]]
    assert ctc_greedy_decode(lp, torch.tensor([5]), blank) == [[0, 0, 1]]
