import torch


class DisCoGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor.

    This implementation is based on the DisCo-CLIP paper and their codebase:
    https://github.com/IDEA-Research/DisCo-CLIP/blob/main/disco/gather.py

    References
    ----------
    DisCo-CLIP: A Distributed Contrastive Loss for Memory Efficient CLIP Training
        Chen, Y., Qi, X., Wang, J., & Zhang, L. (2023).
        In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
        GitHub: https://github.com/IDEA-Research/DisCo-CLIP
    """

    @staticmethod
    def forward(ctx, tensor):
        if not torch.distributed.is_initialized():
            raise ValueError("torch.distributed is not initialized")

        world_size = torch.distributed.get_world_size()
        ctx.bs = tensor.shape[0]
        ctx.rank = torch.distributed.get_rank()

        gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tensors, tensor)

        gathered_tensors = torch.cat(gathered_tensors, dim=0)
        gathered_tensors.requires_grad_(True)

        return gathered_tensors

    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.AVG)
        return grad_output[ctx.bs * ctx.rank : ctx.bs * (ctx.rank + 1)]


def Gather(tensor):
    return DisCoGather.apply(tensor)
