
import torch
import pytest
from src.models.encoder import SemanticEncoder

@pytest.fixture
def encoder():
    return SemanticEncoder(model_name="roberta-base", hidden_dim=512)

def test_encoder_output_shape(encoder):
    """验证编码器输出维度是否符合预期 (hidden_dim)"""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    encoder.eval()
    with torch.no_grad():
        z = encoder(input_ids, attention_mask)
    
    assert z.shape == (batch_size, 512)

def test_roberta_parameters_frozen(encoder):
    """验证 RoBERTa 骨干网络的参数已被冻结"""
    for name, param in encoder.roberta.named_parameters():
        assert not param.requires_grad, f"Parameter {name} should be frozen"

def test_adapter_parameters_trainable(encoder):
    """验证 Adapter 和 VIB 层的参数是可训练的"""
    for name, param in encoder.adapter.named_parameters():
        assert param.requires_grad, f"Adapter parameter {name} should be trainable"
    
    assert encoder.mu_layer.weight.requires_grad
    assert encoder.logvar_layer.weight.requires_grad

def test_kl_divergence_return(encoder):
    """验证在训练模式下返回 KL 散度"""
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 50265, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    
    encoder.train()
    z, kl_loss = encoder(input_ids, attention_mask, return_kl=True)
    
    assert z.shape == (batch_size, 512)
    assert kl_loss.shape == (batch_size,)
    assert (kl_loss >= 0).all(), "KL loss should be non-negative"

def test_encode_text_convenience_method(encoder):
    """验证 encode_text 便捷方法的可用性"""
    text = "Hello, this is a test sentence."
    z = encoder.encode_text(text)
    
    assert z.shape == (1, 512)
    assert isinstance(z, torch.Tensor)
