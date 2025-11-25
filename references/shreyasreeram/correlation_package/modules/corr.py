from torch.nn.modules.module import Module


from torch.nn.modules.module import Module

# Define Correlation class
class Correlation(Module):
    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None,
                 stride1=None, stride2=None, corr_multiply=None):
        super(Correlation, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        # Import correlation function here to avoid circular import issues
        from correlation_package.functions.corr import correlation
        return correlation(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)

    def __repr__(self):
        return self.__class__.__name__

# Define Correlation1d class
class Correlation1d(Module):
    def __init__(self, pad_size=None, kernel_size=None, max_displacement=None,
                 stride1=None, stride2=None, corr_multiply=None):
        super(Correlation1d, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply

    def reset_params(self):
        return

    def forward(self, input1, input2):
        # Import correlation1d function here to avoid circular import issues
        from correlation_package.functions.corr import correlation1d
        return correlation1d(self.pad_size, self.kernel_size, self.max_displacement, self.stride1, self.stride2, self.corr_multiply)(input1, input2)

    def __repr__(self):
        return self.__class__.__name__