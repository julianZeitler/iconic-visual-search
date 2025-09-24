import numpy as np
import cv2
from scipy.signal import convolve2d
from typing import List
import warnings
warnings.filterwarnings("ignore")


class SpatioChromaticFilters:
    """
    Implementation of the steerable filters described in the paper.
    Uses Gaussian derivatives up to 3rd order as basis functions.
    """

    def __init__(self, filter_size: int = 21):
        self.filter_size = filter_size
        self.filters = self._create_basis_filters()

    def _create_basis_filters(self) -> List[np.ndarray]:
        """Create the 10 basis filters: G0, G1 (2 filters), G2 (3 filters), G3 (4 filters)"""
        filters = []
        center = self.filter_size // 2
        x, y = np.meshgrid(np.arange(self.filter_size) - center,
                          np.arange(self.filter_size) - center)

        # G0 - Gaussian
        sigma = self.filter_size / 6.0
        gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        filters.append(gaussian)

        # G1 - First derivatives (2 filters)
        g1_x = -x * gaussian / sigma**2
        g1_y = -y * gaussian / sigma**2
        filters.extend([g1_x, g1_y])

        # G2 - Second derivatives (3 filters)
        g2_xx = ((x**2 / sigma**2) - 1) * gaussian / sigma**2
        g2_yy = ((y**2 / sigma**2) - 1) * gaussian / sigma**2
        g2_xy = (x * y * gaussian) / sigma**4
        filters.extend([g2_xx, g2_yy, g2_xy])

        # G3 - Third derivatives (4 filters)
        g3_xxx = x * ((x**2 / sigma**2) - 3) * gaussian / sigma**4
        g3_yyy = y * ((y**2 / sigma**2) - 3) * gaussian / sigma**4
        g3_xxy = y * ((x**2 / sigma**2) - 1) * gaussian / sigma**4
        g3_xyy = x * ((y**2 / sigma**2) - 1) * gaussian / sigma**4
        filters.extend([g3_xxx, g3_yyy, g3_xxy, g3_xyy])

        return filters

    def apply_filters(self, image: np.ndarray, scales: List[float]) -> np.ndarray:
        """
        Apply all filters at multiple scales to create iconic representation
        Returns: filter_responses of shape (height, width, num_filters * num_scales)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image = image.astype(np.float32) / 255.0
        height, width = image.shape
        num_filters = len(self.filters)
        num_scales = len(scales)

        responses = np.zeros((height, width, num_filters * num_scales))

        for scale_idx, scale in enumerate(scales):
            # Create scaled filters
            scaled_filters = []
            for filter_kernel in self.filters:
                if scale != 1.0:
                    # Scale the filter by resizing
                    new_size = max(3, int(self.filter_size * scale))
                    if new_size % 2 == 0:
                        new_size += 1
                    scaled_filter = cv2.resize(filter_kernel, (new_size, new_size))
                    # Normalize
                    if np.sum(np.abs(scaled_filter)) > 0:
                        scaled_filter = scaled_filter / np.sum(np.abs(scaled_filter))
                else:
                    scaled_filter = filter_kernel
                scaled_filters.append(scaled_filter)

            # Apply each scaled filter
            for filter_idx, scaled_filter in enumerate(scaled_filters):
                response = convolve2d(image, scaled_filter, mode='same', boundary='symm')
                response_idx = scale_idx * num_filters + filter_idx
                responses[:, :, response_idx] = response

        return responses