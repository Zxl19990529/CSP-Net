class DenseCRFParams(object):
    """
    Parameters for the DenseCRF model
    """

    def __init__(self, alpha=80, beta=13, gamma=3, spatial_ker_weight=3, bilateral_ker_weight=10):
        """
        Default values were taken from the original DenseCRF code in
        https://www.philkr.net/papers/2011-12-01-nips/densecrf_v_2_2.zip

        Args:
            alpha: sxy_Bilateral 80
            beta: srgb 13
            gamma: sxy_Gaussian 3
            spatial_ker_weight: compat_Gaussian 3
            bilateral_ker_weight: compat_Bilateral 10
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spatial_ker_weight = spatial_ker_weight
        self.bilateral_ker_weight = bilateral_ker_weight
