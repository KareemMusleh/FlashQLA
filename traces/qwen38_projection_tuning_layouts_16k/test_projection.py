"""cuBLASLt ABI/layout correctness against independent FP64 matmul."""
import pytest
import torch
from flash_qla.ops.gated_delta_rule.chunk.hopper.projection import ProjectionPlan


@pytest.mark.parametrize('transpose',[False,True])
@pytest.mark.parametrize('column',[False,True])
@torch.no_grad()
def test_projection(transpose,column):
    torch.manual_seed(5)
    x=torch.randn(67,128,device='cuda',dtype=torch.bfloat16)
    w=torch.randn(96,128,device='cuda',dtype=torch.bfloat16)
    plan=ProjectionPlan(67,96,128,x.device,weight_transposed=transpose,column_order=column,algorithms=4)
    try:
        actual=plan.run(x,w.T.contiguous() if transpose else w)
        expected=(x.double()@w.double().T).bfloat16()
        torch.testing.assert_close(actual,expected,rtol=.008,atol=.008)
        assert (actual.float()-expected.float()).norm()/expected.float().norm()<.001
        torch.cuda.synchronize()
    finally:plan.close()
