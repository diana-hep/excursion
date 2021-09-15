from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel


def get_kernel(ndim, kernel_type):
    if kernel_type == 'const_rbf':
        length_scale = [1.]*ndim
        kernel = ConstantKernel() * RBF(length_scale_bounds=[0.1, 100.0], length_scale=length_scale)
    elif kernel_type == 'tworbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + \
                 ConstantKernel() * RBF(length_scale_bounds=[100., 1000.0]) + \
                 WhiteKernel(noise_level_bounds=[1e-7, 1e-4])
    elif kernel_type == 'onerbf_white':
        kernel = ConstantKernel() * RBF(length_scale_bounds=[1e-2, 100]) + WhiteKernel(noise_level_bounds=[1e-7, 1e-1])
    else:
        raise RuntimeError('unknown kernel')
    return kernel