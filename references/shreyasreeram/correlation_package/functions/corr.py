import torch
from torch.autograd import Function

# Import correlation CUDA functions from _ext
from correlation_package._ext import corr  # Ensure this path is correct

class CorrelationFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        # Save parameters for backward pass
        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        # Initialize output tensor
        output = torch.empty_like(input1)  # Adjust shape if necessary

        # Call CUDA forward function
        corr.corr_cuda_forward(input1, input2, output, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        pad_size = ctx.pad_size
        kernel_size = ctx.kernel_size
        max_displacement = ctx.max_displacement
        stride1 = ctx.stride1
        stride2 = ctx.stride2
        corr_multiply = ctx.corr_multiply

        # Initialize gradients for inputs
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)

        # Call CUDA backward function
        corr.corr_cuda_backward(input1, input2, grad_output, grad_input1, grad_input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)
        return grad_input1, grad_input2, None, None, None, None, None, None

# Alias for easier use in other files
correlation = CorrelationFunction.apply

class Correlation1d(Function):
    @staticmethod
    def forward(ctx, input1, input2, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=1, corr_multiply=1):
        from .._ext import corr  # Import inside the function

        ctx.save_for_backward(input1, input2)
        ctx.pad_size = pad_size
        ctx.kernel_size = kernel_size
        ctx.max_displacement = max_displacement
        ctx.stride1 = stride1
        ctx.stride2 = stride2
        ctx.corr_multiply = corr_multiply

        rbot1 = input1.new()
        rbot2 = input2.new()
        output = input1.new()

        corr.corr1d_cuda_forward(input1, input2,
                                 rbot1, rbot2,
                                 output,
                                 pad_size,
                                 kernel_size,
                                 max_displacement,
                                 stride1,
                                 stride2,
                                 corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        from .._ext import corr  # Import inside the function

        input1, input2 = ctx.saved_tensors
        pad_size = ctx.pad_size
        kernel_size = ctx.kernel_size
        max_displacement = ctx.max_displacement
        stride1 = ctx.stride1
        stride2 = ctx.stride2
        corr_multiply = ctx.corr_multiply

        rbot1 = input1.new()
        rbot2 = input2.new()
        grad_input1 = torch.zeros_like(input1).cuda()
        grad_input2 = torch.zeros_like(input2).cuda()

        corr.corr1d_cuda_backward(input1, input2,
                                  rbot1, rbot2,
                                  grad_output,
                                  grad_input1,
                                  grad_input2,
                                  pad_size,
                                  kernel_size,
                                  max_displacement,
                                  stride1,
                                  stride2,
                                  corr_multiply)

        return grad_input1, grad_input2, None, None, None, None, None, None