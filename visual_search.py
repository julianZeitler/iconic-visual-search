import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from spatio_chromatic_filters import SpatioChromaticFilters


class VisualSearchModel:
    """
    Implementation of the visual search model from the paper.
    Uses coarse-to-fine strategy with weighted population averaging.
    """

    def __init__(self, scales: List[float] = [2.0, 1.0, 0.5]):
        self.scales = scales
        self.filter_bank = SpatioChromaticFilters()
        self.target_template = None
        self.target_bbox = None
        self.temperature_schedule = [1.0, 0.1, 0.01]

    def memorize_target(self, target_image: np.ndarray, bbox: dict):
        """
        Memorize target using bounding box from annotations.
        bbox: dictionary with keys 'x1', 'y1', 'x2', 'y2' defining the target region
        """
        # Extract target region using bounding box coordinates
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])

        # Ensure coordinates are within image bounds
        height, width = target_image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        # Ensure we have a valid bounding box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Calculate target center coordinates (x_t, y_t) as in the paper
        target_center_x = (x1 + x2) // 2
        target_center_y = (y1 + y2) // 2

        # Get filter responses for entire image
        image_responses = self.filter_bank.apply_filters(target_image, self.scales)

        # Extract response vector at target center (x_t, y_t) as in the paper
        self.target_template = image_responses[target_center_y, target_center_x, :]
        self.target_bbox = bbox

    def compute_saliency_map(self, scene_responses: np.ndarray, current_scale_level: int) -> np.ndarray:
        """
        Compute saliency map using scales from coarsest to current_scale_level.
        As described in the paper: start with coarsest scale, progressively add finer scales.
        current_scale_level: 0=first fixation (coarsest only), 2=final fixation (all scales)
        scene_responses: Pre-computed filter responses for the entire image
        """
        if self.target_template is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        height, width = scene_responses.shape[:2]

        # Use scales from coarsest (0) up to and including current_scale_level
        # Fixation 1: scales [0] (coarsest only)
        # Fixation 2: scales [0, 1] (coarsest + next)
        # Fixation 3: scales [0, 1, 2] (all scales)
        num_filters = len(self.filter_bank.filters)

        saliency_map = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                distance = 0
                # Integrate from coarsest scale (0) to current scale level
                for scale_idx in range(current_scale_level + 1):
                    for filter_idx in range(num_filters):
                        response_idx = scale_idx * num_filters + filter_idx
                        scene_response = scene_responses[y, x, response_idx]
                        target_response = self.target_template[response_idx]
                        distance += (scene_response - target_response) ** 2

                saliency_map[y, x] = distance

        return saliency_map

    def weighted_population_averaging(self, saliency_map: np.ndarray, lambda_k: float) -> Tuple[int, int]:
        """
        Compute next fixation using weighted population averaging (Equations 7 and 8).

        Args:
            saliency_map: Saliency values S(x,y) from equation 6
            lambda_k: Temperature parameter λ(k) for the current iteration

        Returns:
            (x, y) coordinates of weighted average fixation location
        """
        height, width = saliency_map.shape

        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        # Compute weights using equation 8: F(S(x,y)) = exp(-S(x,y)/λ(k))
        # Note: negative sign because lower saliency (better match) should have higher weight
        weights = np.exp(-saliency_map / lambda_k)

        # Normalize weights to sum to 1
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            normalized_weights = weights / weight_sum
        else:
            # Fallback: uniform weights if all weights are zero
            normalized_weights = np.ones_like(weights) / (height * width)

        # Compute weighted average position (equation 7)
        x_target = np.sum(normalized_weights * x_coords)
        y_target = np.sum(normalized_weights * y_coords)

        return int(round(x_target)), int(round(y_target))

    def visual_search(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Perform coarse-to-fine visual search returning sequence of fixations.
        Returns list of (x, y) fixation points.
        """
        if self.target_template is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        # Compute filter responses for entire image once (optimization)
        scene_responses = self.filter_bank.apply_filters(image, self.scales)

        fixations = []

        # Coarse-to-fine search
        for scale_level in range(len(self.scales)):
            # Compute saliency map at current scale level using pre-computed responses
            saliency_map = self.compute_saliency_map(scene_responses, scale_level)

            # Get temperature parameter λ(k) for current iteration
            lambda_k = self.temperature_schedule[scale_level]

            # Compute fixation point using weighted population averaging (equations 7 and 8)
            fixation_x, fixation_y = self.weighted_population_averaging(saliency_map, lambda_k)

            fixations.append((fixation_x, fixation_y))

        return fixations

    def visualize_search(self, image: np.ndarray, fixations: List[Tuple[int, int]],
                        target_location: Optional[Tuple[int, int]] = None):
        """Visualize the search process with fixation sequence."""
        plt.figure(figsize=(8, 6))

        # Show image with fixation sequence
        if len(image.shape) == 3:
            plt.imshow(image)
        else:
            plt.imshow(image, cmap='gray')

        # Plot fixations with arrows
        for i, (x, y) in enumerate(fixations):
            plt.plot(x, y, 'ro', markersize=8)
            plt.text(x+5, y+5, f'{i+1}', color='red', fontweight='bold')
            if i > 0:
                prev_x, prev_y = fixations[i-1]
                plt.arrow(prev_x, prev_y, x-prev_x, y-prev_y,
                         head_width=5, head_length=8, fc='red', ec='red')

        if target_location:
            plt.plot(target_location[0], target_location[1], 'gs',
                    markersize=12, label='True Target')
            plt.legend()

        plt.title('Visual Search: Fixation Sequence')
        plt.tight_layout()
        plt.show()

    def visualize_filter_responses(self, image: np.ndarray):
        """Visualize filter responses for target patch and full image."""
        if self.target_template is None or self.target_bbox is None:
            print("No target template or bounding box available for filter visualization")
            return

        # Use stored bounding box to extract target patch
        x1, y1 = int(self.target_bbox['x1']), int(self.target_bbox['y1'])
        x2, y2 = int(self.target_bbox['x2']), int(self.target_bbox['y2'])

        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        target_patch = image[y1:y2, x1:x2]

        # Get filter responses for target patch and full image
        target_responses = self.filter_bank.apply_filters(target_patch, self.scales)
        image_responses = self.filter_bank.apply_filters(image, self.scales)

        # Select example filters to show (one from each derivative order)
        example_filters = [
            (0, 'G0 - Gaussian'),           # Scale 0, Filter 0 (Gaussian)
            (1, 'G1x - 1st derivative X'),  # Scale 0, Filter 1 (G1x)
            (3, 'G2xx - 2nd derivative XX'), # Scale 0, Filter 3 (G2xx)
            (6, 'G3xxx - 3rd derivative XXX') # Scale 0, Filter 6 (G3xxx)
        ]

        n_filters = len(example_filters)
        n_scales = len(self.scales)

        # Create figure with better spacing
        fig = plt.figure(figsize=(24, 16))

        # Use gridspec for better control
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(n_filters * 2, n_scales + 2, figure=fig,
                     wspace=0.4, hspace=0.6,
                     left=0.05, right=0.95, top=0.92, bottom=0.08)

        for filter_idx, (base_filter_idx, filter_name) in enumerate(example_filters):
            # Show target patch (leftmost column) - spans both rows
            ax_patch = fig.add_subplot(gs[filter_idx*2:(filter_idx+1)*2, 0])
            if len(target_patch.shape) == 3:
                ax_patch.imshow(target_patch)
            else:
                ax_patch.imshow(target_patch, cmap='gray')
            ax_patch.set_title(f'Target Patch\n{filter_name}', fontsize=10, pad=10)
            ax_patch.axis('off')

            # Show filter kernel (second column) - spans both rows
            ax_kernel = fig.add_subplot(gs[filter_idx*2:(filter_idx+1)*2, 1])
            kernel = self.filter_bank.filters[base_filter_idx]
            im_kernel = ax_kernel.imshow(kernel, cmap='RdBu_r',
                                       vmin=-np.max(np.abs(kernel)),
                                       vmax=np.max(np.abs(kernel)))
            ax_kernel.set_title('Filter Kernel', fontsize=10, pad=10)
            ax_kernel.axis('off')
            cbar_kernel = plt.colorbar(im_kernel, ax=ax_kernel, shrink=0.8, aspect=30)
            cbar_kernel.ax.tick_params(labelsize=8)

            # Show responses at different scales
            for scale_idx in range(n_scales):
                response_idx = scale_idx * len(self.filter_bank.filters) + base_filter_idx
                target_response = target_responses[:, :, response_idx]

                # Top half: target patch response
                ax_top = fig.add_subplot(gs[filter_idx * 2, scale_idx + 2])
                im1 = ax_top.imshow(target_response, cmap='RdBu_r')
                ax_top.set_title(f'Target Response\nScale {self.scales[scale_idx]:.1f}',
                               fontsize=9, pad=8)
                ax_top.axis('off')
                cbar1 = plt.colorbar(im1, ax=ax_top, shrink=0.8, aspect=15)
                cbar1.ax.tick_params(labelsize=7)

                # Bottom half: full image response
                ax_bottom = fig.add_subplot(gs[filter_idx * 2 + 1, scale_idx + 2])

                # Show full image response
                image_response = image_responses[:, :, response_idx]

                im2 = ax_bottom.imshow(image_response, cmap='RdBu_r')
                ax_bottom.set_title('Image Response', fontsize=9, pad=8)
                ax_bottom.axis('off')

                # Mark target center in full image response
                target_center_x = (x1 + x2) // 2
                target_center_y = (y1 + y2) // 2
                ax_bottom.plot(target_center_x, target_center_y, 'g*', markersize=8)

                cbar2 = plt.colorbar(im2, ax=ax_bottom, shrink=0.8, aspect=15)
                cbar2.ax.tick_params(labelsize=7)

        plt.suptitle('Filter Responses: Target vs Image', fontsize=16, y=0.96)
        plt.show()

    def visualize_consecutive_saliency_maps(self, image: np.ndarray, fixations: List[Tuple[int, int]],
                                          target_location: Optional[Tuple[int, int]] = None):
        """Visualize all consecutive saliency maps for each fixation."""
        # Compute filter responses once
        scene_responses = self.filter_bank.apply_filters(image, self.scales)

        n_scales = len(self.scales)

        # Create figure with subplots for all saliency maps
        fig, axes = plt.subplots(2, n_scales, figsize=(5 * n_scales, 10))
        if n_scales == 1:
            axes = axes.reshape(-1, 1)

        # Top row: saliency maps
        for scale_level in range(n_scales):
            saliency_map = self.compute_saliency_map(scene_responses, scale_level)

            # Invert saliency for visualization: high values = good matches
            saliency_inverted = np.max(saliency_map) - saliency_map

            ax = axes[0, scale_level]
            im = ax.imshow(saliency_inverted, cmap='hot')

            # Show fixation point for this scale level
            if scale_level < len(fixations):
                fix_x, fix_y = fixations[scale_level]
                ax.plot(fix_x, fix_y, 'wo', markersize=10, markeredgecolor='black',
                       markeredgewidth=2, label=f'Fixation {scale_level + 1}')

            # Show target location if provided
            if target_location:
                ax.plot(target_location[0], target_location[1], 'g*',
                       markersize=15, label='Target')

            # Create title based on scales used
            scales_used = self.scales[:scale_level + 1]
            scale_names = [f'{s:.1f}' for s in scales_used]
            ax.set_title(f'Fixation {scale_level + 1}\nScales: {", ".join(scale_names)}',
                        fontsize=12, pad=10)
            ax.legend(loc='upper right')

            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Bottom row: original image
        for scale_level in range(n_scales):
            ax = axes[1, scale_level]

            # Show original image
            if len(image.shape) == 3:
                ax.imshow(image, alpha=1)
            else:
                ax.imshow(image, cmap='gray', alpha=1)

            # Show fixation point for this scale level
            if scale_level < len(fixations):
                fix_x, fix_y = fixations[scale_level]
                ax.plot(fix_x, fix_y, 'wo', markersize=10, markeredgecolor='black',
                       markeredgewidth=2, label=f'Fixation {scale_level + 1}')

            # Show target location if provided
            if target_location:
                ax.plot(target_location[0], target_location[1], 'g*',
                       markersize=15, label='Target')

            ax.set_title(f'Saliency Overlay - Fixation {scale_level + 1}',
                        fontsize=12, pad=10)
            ax.legend(loc='upper right')
            ax.axis('off')

        plt.suptitle('Consecutive Saliency Maps: Coarse-to-Fine Progression',
                     fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()
