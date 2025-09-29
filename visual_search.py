import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from spatio_chromatic_filters import SpatioChromaticFilters
from gist.layers.gabor_filter_bank import GaborFilterbank


class VisualSearchModel:
    """
    Implementation of the visual search model from the paper.
    Uses coarse-to-fine strategy with weighted population averaging.
    """

    def __init__(self, scales: List[float] = [2.0, 1.0, 0.5], device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.scales = scales
        self.device = torch.device(device)
        self.filter_bank = GaborFilterbank(
            in_channels=1,
            mode="static",
            bandwidth=0.1,
            theta=np.pi/5,
            n_scales=3,
            n_orientations=8,
            n_phases=2,
            fmax=0.3,
            scale=5,
            gaussian=True,
            gaussian_inverse=False,
            n_stds=3,
            dc_compensate=True,
            stride=1,
            energy=True,
            energy_mode="substitute"
        ).to(self.device)
        self.target_template = None
        self.target_bbox = None
        self.temperature_schedule = [0.01, 0.005, 0.001]

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to PyTorch tensor with proper shape and normalization."""
        if len(image.shape) == 2:
            # Grayscale image
            tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # Grayscale image with channel dimension
            tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
        elif len(image.shape) == 3:
            # RGB image - convert to grayscale
            if image.shape[2] == 3:
                # RGB to grayscale conversion
                gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
                tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
            else:
                tensor = torch.from_numpy(image).float().unsqueeze(0)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return tensor.to(self.device)

    def _apply_filters(self, image: np.ndarray) -> torch.Tensor:
        """Apply Gabor filter bank to image using PyTorch."""
        tensor = self._image_to_tensor(image)

        # Apply filter bank (already includes multiple scales)
        with torch.no_grad():
            responses = self.filter_bank(tensor)

        # Remove batch dimension and permute to (H, W, C)
        return responses.squeeze(0).permute(1, 2, 0)

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

        # Get filter responses for entire image using PyTorch
        image_responses = self._apply_filters(target_image)

        # Extract response vector at target center (x_t, y_t) as in the paper
        self.target_template = image_responses[target_center_y, target_center_x, :]
        self.target_bbox = bbox

    def compute_saliency_map(self, scene_responses: torch.Tensor, current_scale_level: int) -> torch.Tensor:
        """
        Compute saliency map using scales from coarsest to current_scale_level.
        As described in the paper: start with coarsest scale, progressively add finer scales.
        current_scale_level: 0=first fixation (coarsest only), 2=final fixation (all scales)
        scene_responses: Pre-computed filter responses for the entire image (PyTorch tensor)
        """
        if self.target_template is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        # Use scales from coarsest up to and including current_scale_level
        # Channel organization: 0=DC, 1-8=coarsest, 9-16=scale2, 17-24=scale3, 25-32=finest
        # Fixation 1: DC + coarsest (channels 0-8)
        # Fixation 2: DC + coarsest + scale2 (channels 0-16)
        # Fixation 3: DC + coarsest + scale2 + scale3 (channels 0-24)

        # Always include DC component (channel 0)
        channels_to_use = [0]

        # Add scale channels based on current_scale_level
        filters_per_scale = 8
        for scale_idx in range(current_scale_level + 1):
            start_idx = 1 + scale_idx * filters_per_scale  # +1 to skip DC component
            end_idx = 1 + (scale_idx + 1) * filters_per_scale
            channels_to_use.extend(range(start_idx, end_idx))

        # Extract relevant channels
        scene_subset = scene_responses[:, :, channels_to_use]  # (H, W, selected_channels)
        target_subset = self.target_template[channels_to_use]  # (selected_channels,)

        # Compute squared differences using broadcasting
        # scene_subset: (H, W, C), target_subset: (C,) -> (H, W, C)
        differences = scene_subset - target_subset.unsqueeze(0).unsqueeze(0)

        # Sum squared differences across channels
        saliency_map = torch.sum(differences ** 2, dim=2)  # (H, W)

        # Normalize the saliency map
        saliency_min = torch.min(saliency_map)
        saliency_max = torch.max(saliency_map)
        if saliency_max > saliency_min:
            saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency_map = torch.zeros_like(saliency_map)

        return saliency_map

    def weighted_population_averaging(self, saliency_map: torch.Tensor, lambda_k: float) -> Tuple[int, int]:
        """
        Compute next fixation using weighted population averaging (Equations 7 and 8).

        Args:
            saliency_map: Saliency values S(x,y) from equation 6 (PyTorch tensor)
            lambda_k: Temperature parameter λ(k) for the current iteration

        Returns:
            (x, y) coordinates of weighted average fixation location
        """
        height, width = saliency_map.shape

        # Create coordinate grids using PyTorch
        x_coords = torch.arange(width, device=self.device, dtype=torch.float32).unsqueeze(0).expand(height, -1)
        y_coords = torch.arange(height, device=self.device, dtype=torch.float32).unsqueeze(1).expand(-1, width)

        # Compute weights using equation 8: F(S(x,y)) = exp(-S(x,y)/λ(k))
        # Note: negative sign because lower saliency (better match) should have higher weight
        weights = torch.exp(-saliency_map / lambda_k)

        # Normalize weights to sum to 1
        weight_sum = torch.sum(weights)
        if weight_sum > 0:
            normalized_weights = weights / weight_sum
        else:
            # Fallback: uniform weights if all weights are zero
            normalized_weights = torch.ones_like(weights) / (height * width)

        # Compute weighted average position (equation 7)
        x_target = torch.sum(normalized_weights * x_coords)
        y_target = torch.sum(normalized_weights * y_coords)

        return int(round(x_target.item())), int(round(y_target.item()))

    def visual_search(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Perform coarse-to-fine visual search returning sequence of fixations.
        Returns list of (x, y) fixation points.
        """
        if self.target_template is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        # Compute filter responses for entire image once (optimization)
        scene_responses = self._apply_filters(image)

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

    def visualize_consecutive_saliency_maps(self, image: np.ndarray, fixations: List[Tuple[int, int]],
                                          target_location: Optional[Tuple[int, int]] = None):
        """Visualize all consecutive saliency maps for each fixation."""
        # Compute filter responses once
        scene_responses = self._apply_filters(image)

        n_scales = len(self.scales)

        # Create figure with subplots for all saliency maps
        fig, axes = plt.subplots(2, n_scales, figsize=(5 * n_scales, 10))
        if n_scales == 1:
            axes = axes.reshape(-1, 1)

        # Top row: saliency maps
        for scale_level in range(n_scales):
            saliency_map = self.compute_saliency_map(scene_responses, scale_level)

            # Convert to numpy for visualization
            saliency_np = saliency_map.cpu().numpy()

            # Invert saliency for visualization: high values = good matches
            saliency_inverted = np.max(saliency_np) - saliency_np

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
