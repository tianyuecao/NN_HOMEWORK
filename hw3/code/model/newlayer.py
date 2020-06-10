from torch.autograd import Function


class GradientReversalLayer(Function):
    """
    Define a new function with forward and backward method.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return - grad_output * ctx.alpha, None
