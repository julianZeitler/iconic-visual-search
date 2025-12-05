import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import VGG16_Weights
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from gist.layers.gabor_filter_bank import GaborFilterbank


class VisualSearchModel:
    """
    Implementation of the visual search model from the paper.
    Uses coarse-to-fine strategy with weighted population averaging.
    """

    def __init__(self, backbone: nn.Module, gist: Optional[nn.Module] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)

        #TODO: Support Gist (NOT supported right now!)
        if gist:
            self.gist = gist.to(self.device)
        else:
            self.gist = None
        
        self.backbone = backbone.to(self.device)

        self.target_template = None
        self.target_variance = None
        self.temperature_schedule = [0.1, 0.05, 0.001]

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_backbone(self, x: torch.Tensor, response_layer: Optional[int] = None, batch_norm: bool = True) -> torch.Tensor:
        """Apply backbone to image. Output is response of layer with idx 'response_layer'."""

        # Ensure image has 4 dimensions (B, C, H, W)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dim

        if self.gist:
            x = self.gist(x)

        with torch.no_grad():
            for i, layer in enumerate(self.backbone.features):
                if response_layer and i > response_layer:
                    break

                if isinstance(layer, nn.BatchNorm2d) and not batch_norm:
                    continue
                elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.InstanceNorm2d):
                    if x.shape[-2] == 1 or x.shape[-1] == 1:
                        # don't apply instance norm for single-dimension feature maps
                        continue
                elif isinstance(layer, nn.AdaptiveAvgPool2d):
                    continue
                elif isinstance(layer, nn.Flatten):
                    continue
                
                x = layer(x)
        return x

    def memorize_target(self, target_image: torch.Tensor, bbox: dict, layer: int = 12):
        """
        Memorize target using bounding box from annotations. Target representation is response in layer at bounding box center.
        bbox: dictionary with keys 'x1', 'y1', 'x2', 'y2' defining the target region
        """
        # Extract target region using bounding box coordinates (in original image space)
        x1, y1 = int(bbox['x1']), int(bbox['y1'])
        x2, y2 = int(bbox['x2']), int(bbox['y2'])

        # Get original image dimensions
        orig_height, orig_width = target_image.shape[-2], target_image.shape[-1]

        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, orig_width - 1))
        x2 = max(0, min(x2, orig_width))
        y1 = max(0, min(y1, orig_height - 1))
        y2 = max(0, min(y2, orig_height))

        # Ensure we have a valid bounding box
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        image_responses = self._apply_backbone(target_image, response_layer=layer, batch_norm=False)

        # Get feature map dimensions (C, H_feat, W_feat)
        shape = image_responses.shape
        feat_height = shape[-2]
        feat_width = shape[-1]

        # Get scaling factors using helper method
        scale_x = feat_width / orig_width
        scale_y = feat_height / orig_height

        # Calculate target center coordinates (x_t, y_t) in original image space
        target_center_x_orig = (x1 + x2) / 2.0
        target_center_y_orig = (y1 + y2) / 2.0

        # Convert center coordinates to feature map space
        target_center_x_feat = target_center_x_orig * scale_x
        target_center_y_feat = target_center_y_orig * scale_y

        # Round to nearest integer and ensure within bounds
        target_center_x_feat = int(round(target_center_x_feat))
        target_center_y_feat = int(round(target_center_y_feat))
        target_center_x_feat = max(0, min(target_center_x_feat, feat_width - 1))
        target_center_y_feat = max(0, min(target_center_y_feat, feat_height - 1))

        # Extract response vector at target center (x_t, y_t) as in the paper
        self.target_template = image_responses[:, :, target_center_y_feat, target_center_x_feat].squeeze(0)
        self.target_variance = torch.zeros_like(self.target_template)

    def memorize_target_batch(self, image_batch: torch.Tensor, bboxes: List[dict], layer: int = 12):
        """
        Memorize target from a batch of object instances.
        Computes mean and variance across all instance responses.
        Processes all images as a single batch for efficiency.

        Args:
            image_batch: Batch of images as PyTorch tensor (B, C, H, W), normalized to [0, 1]
            bboxes: List of bounding box dictionaries with keys 'x1', 'y1', 'x2', 'y2'
                   (coordinates in transformed/resized image space, matching image_batch dimensions)
            layer: Layer index to extract features from (default: 12)
        """
        if image_batch.size(0) == 0:
            raise ValueError("Cannot memorize from empty batch")

        if len(bboxes) != image_batch.size(0):
            raise ValueError(f"Number of bounding boxes ({len(bboxes)}) must match batch size ({image_batch.size(0)})")

        if image_batch.device != self.device:
            image_batch = image_batch.to(self.device)

        with torch.no_grad():
            batch_responses = self._apply_backbone(image_batch, response_layer=layer, batch_norm=False)

        _, _, feat_height, feat_width = batch_responses.shape
        _, _, img_height, img_width = image_batch.shape

        # Compute scaling from transformed image space to feature map space
        scale_x = feat_width / img_width
        scale_y = feat_height / img_height

        # Extract response vectors for each instance at its target center
        response_vectors = []

        for batch_idx, bbox in enumerate(bboxes):
            # Get bounding box coordinates (already in transformed image space)
            x1, y1 = bbox['x1'], bbox['y1']
            x2, y2 = bbox['x2'], bbox['y2']

            # Calculate target center in transformed image space
            center_x_img = (x1 + x2) / 2.0
            center_y_img = (y1 + y2) / 2.0

            # Convert to feature map space
            center_x_feat = center_x_img * scale_x
            center_y_feat = center_y_img * scale_y

            # Round and clamp to valid indices
            center_x_feat = int(round(center_x_feat))
            center_y_feat = int(round(center_y_feat))
            center_x_feat = max(0, min(center_x_feat, feat_width - 1))
            center_y_feat = max(0, min(center_y_feat, feat_height - 1))

            # Extract response vector at target center
            # batch_responses is (B, C, H, W), index as [batch, channels, y, x]
            response_vector = batch_responses[batch_idx, :, center_y_feat, center_x_feat]
            response_vectors.append(response_vector)

        # Stack all response vectors: (num_instances, num_channels)
        stacked_responses = torch.stack(response_vectors, dim=0)

        # Compute mean and variance across all instances
        mean_response = torch.mean(stacked_responses, dim=0)
        variance_response = torch.var(stacked_responses, dim=0)

        # Store both mean and variance as the target template
        self.target_template = mean_response
        self.target_variance = variance_response

        print(f"Memorized target from {len(response_vectors)} instances")
        print(f"Response mean range: [{mean_response.min():.4f}, {mean_response.max():.4f}]")
        print(f"Response variance range: [{variance_response.min():.4f}, {variance_response.max():.4f}]")

    def compute_saliency_map(self, scene_responses: torch.Tensor) -> torch.Tensor:
        """
        Compute saliency map
        """
        if self.target_template is None or self.target_variance is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        # Compute squared differences using broadcasting
        differences = scene_responses - self.target_template[None, :, None, None]
        scaled_differences = differences/(1+self.target_variance[None, :, None, None])
        # Sum squared differences across channels
        saliency_map = torch.sum(scaled_differences ** 2, dim=1)  # (B, H, W)

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

    def visual_search(self, image: torch.Tensor, layer: int = 29) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Perform coarse-to-fine visual search returning sequence of fixations.
        Returns list of (x, y) fixation points in original image coordinates.
        """
        if self.target_template is None:
            raise ValueError("No target template stored. Call memorize_target first.")

        scene_responses = self._apply_backbone(image, response_layer=layer)

        orig_height, orig_width = image.shape[-2], image.shape[-1]

        fixations = []
        saliency_map = self.compute_saliency_map(scene_responses)  # (B, H, W)

        # Interpolate saliency map to original image size
        # saliency_map already has batch dimension from compute_saliency_map
        saliency_map_upsampled = torch.nn.functional.interpolate(
            saliency_map.unsqueeze(1),  # Add channel dim: (B, 1, H, W)
            size=(orig_height, orig_width),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # Remove channel dim: (B, H, W)

        for fixation in range(len(self.temperature_schedule)):
            lambda_k = self.temperature_schedule[fixation]
            # saliency_map_upsampled has batch dimension, weighted_population_averaging expects (B, H, W)
            fixation_x_img, fixation_y_img = self.weighted_population_averaging(saliency_map_upsampled.squeeze(0), lambda_k)

            fixation_x_img = max(0, min(fixation_x_img, orig_width - 1))
            fixation_y_img = max(0, min(fixation_y_img, orig_height - 1))

            fixations.append((fixation_x_img, fixation_y_img))

        # Return first batch element (squeeze batch dimension for single image)
        return saliency_map, saliency_map_upsampled, fixations

    def visualize_search(self, image: torch.Tensor, saliency_map: torch.Tensor, fixations: List[Tuple[int, int]],
                        target_location: Optional[Tuple[int, int]] = None):
        """
        Visualize the visual search process.

        Args:
            image: Original image (numpy array)
            saliency_map: Saliency map (PyTorch tensor, same size as image)
            fixations: List of (x, y) fixation coordinates
            target_location: Optional ground truth target location (x, y)
        """
        np_image = image.cpu().numpy().transpose(1, 2, 0)
        # Convert saliency map to numpy for visualization
        if isinstance(saliency_map, torch.Tensor):
            saliency_map_np = saliency_map.cpu().numpy()
        else:
            saliency_map_np = saliency_map

        saliency_inverted = np.max(saliency_map_np) - saliency_map_np

        # Create figure with three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Subplot 1: Original image
        if len(image.shape) == 3:
            axes[0].imshow(np_image)
        else:
            axes[0].imshow(np_image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # Subplot 2: Saliency map
        im = axes[1].imshow(saliency_inverted, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Inverted Saliency Map', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Saliency', rotation=270, labelpad=15)

        # Subplot 3: Image with saliency overlay and fixations
        if len(np_image.shape) == 3:
            axes[2].imshow(np_image)
        else:
            axes[2].imshow(np_image, cmap='gray')

        # Overlay saliency map with transparency
        axes[2].imshow(saliency_inverted, cmap='hot', alpha=0.4, interpolation='bilinear')

        # Plot fixation sequence with arrows
        for i, (x, y) in enumerate(fixations):
            # Plot fixation point
            axes[2].plot(x, y, 'co', markersize=10, markeredgewidth=2, markeredgecolor='white')

            # Add fixation number
            axes[2].text(x + 8, y + 8, f'{i+1}', color='cyan', fontweight='bold',
                        fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

            # Draw arrow from previous fixation
            if i > 0:
                prev_x, prev_y = fixations[i-1]
                axes[2].annotate('', xy=(x, y), xytext=(prev_x, prev_y),
                               arrowprops=dict(arrowstyle='->', color='cyan', lw=2,
                                             connectionstyle='arc3,rad=0.1'))

        # Plot ground truth target location if provided
        if target_location:
            tx, ty = target_location
            axes[2].plot(tx, ty, 'g*', markersize=20, markeredgewidth=2,
                        markeredgecolor='white', label='Ground Truth Target')
            axes[2].legend(loc='upper right', fontsize=10)

        axes[2].set_title('Image + Saliency + Fixations', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
        