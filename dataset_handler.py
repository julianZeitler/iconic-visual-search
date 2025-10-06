import os
import numpy as np
from PIL import Image
from torchvision.datasets import CocoDetection
from typing import List, Dict, Tuple, Optional
import random


class COCODatasetHandler:
    """
    Handler for loading and processing COCO dataset using torchvision.
    Provides functionality to find instances of specified classes.
    """

    def __init__(self, data_dir: str, data_type: str = "val2017"):
        """
        Initialize COCO dataset handler.

        Args:
            data_dir: Root directory of COCO dataset
            data_type: Dataset type (e.g., 'train2017', 'val2017')
        """
        self.data_dir = data_dir
        self.data_type = data_type

        img_dir = os.path.join(data_dir, data_type)
        ann_file = os.path.join(data_dir, f'annotations/instances_{data_type}.json')

        # Initialize COCO dataset
        self.dataset = CocoDetection(root=img_dir, annFile=ann_file)
        self.coco = self.dataset.coco

        # Get category information
        self.categories = self.coco.loadCats(self.coco.getCatIds())
        self.category_names = [cat['name'] for cat in self.categories]

    def get_category_id(self, class_name: str) -> Optional[int]:
        """
        Get category ID from class name.

        Args:
            class_name: Name of the class (e.g., 'person', 'car')

        Returns:
            Category ID or None if not found
        """
        cat_ids = self.coco.getCatIds(catNms=[class_name])
        return cat_ids[0] if cat_ids else None

    def find_instances(self, class_name: str, num_instances: int,
                       min_area: Optional[float] = None,
                       max_area: Optional[float] = None,
                       shuffle: bool = False) -> List[Dict]:
        """
        Find a specified number of instances of a given class.

        Args:
            class_name: Name of the class to find (e.g., 'person', 'car')
            num_instances: Number of instances to retrieve
            min_area: Minimum area of bounding box (optional filter)
            max_area: Maximum area of bounding box (optional filter)
            shuffle: Whether to randomly shuffle results

        Returns:
            List of dictionaries containing instance information:
            - 'image_id': COCO image ID
            - 'image_path': Full path to image file
            - 'annotation_id': Annotation ID
            - 'bbox': Bounding box in COCO format [x, y, width, height]
            - 'bbox_dict': Bounding box as dict with keys 'x1', 'y1', 'x2', 'y2'
            - 'area': Area of the bounding box
            - 'category_id': Category ID
            - 'category_name': Category name
            - 'image': PIL Image object
        """
        # Get category ID
        cat_id = self.get_category_id(class_name)
        if cat_id is None:
            raise ValueError(f"Class '{class_name}' not found in COCO dataset. "
                           f"Available classes: {', '.join(self.category_names)}")

        # Get all image IDs containing this category
        img_ids = self.coco.getImgIds(catIds=[cat_id])

        if shuffle:
            random.shuffle(img_ids)

        instances = []

        # Iterate through images until we have enough instances
        for img_id in img_ids:
            if len(instances) >= num_instances:
                break

            # Get annotations for this image
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[cat_id])
            anns = self.coco.loadAnns(ann_ids)

            # Get image info and load image
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.dataset.root, img_info['file_name'])
            img = Image.open(img_path).convert('L')

            # Process each annotation
            for ann in anns:
                if len(instances) >= num_instances:
                    break

                # Apply area filters if specified
                if min_area is not None and ann['area'] < min_area:
                    continue
                if max_area is not None and ann['area'] > max_area:
                    continue

                # Convert bbox from COCO format [x, y, width, height] to dict format
                x, y, w, h = ann['bbox']
                bbox_dict = {
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h
                }

                instance = {
                    'image_id': img_id,
                    'image_path': img_path,
                    'annotation_id': ann['id'],
                    'bbox': ann['bbox'],
                    'bbox_dict': bbox_dict,
                    'area': ann['area'],
                    'category_id': cat_id,
                    'category_name': class_name,
                    'image': img
                }

                instances.append(instance)

        if len(instances) < num_instances:
            print(f"Warning: Only found {len(instances)} instances of '{class_name}', "
                  f"requested {num_instances}")

        return instances[:num_instances]

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path and convert to numpy array.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (RGB format)
        """
        img = Image.open(image_path).convert('RGB')
        return np.array(img)

    def get_all_categories(self) -> List[str]:
        """Get list of all available category names."""
        return self.category_names

    def visualize_instance(self, instance: Dict, show_bbox: bool = True):
        """
        Visualize a single instance with its bounding box.

        Args:
            instance: Instance dictionary from find_instances()
            show_bbox: Whether to draw bounding box
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        # Load image
        if 'image' in instance:
            img = np.array(instance['image'])
        else:
            img = self.load_image(instance['image_path'])

        # Create figure
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img)

        # Draw bounding box if requested
        if show_bbox:
            bbox = instance['bbox_dict']
            rect = patches.Rectangle(
                (bbox['x1'], bbox['y1']),
                bbox['x2'] - bbox['x1'],
                bbox['y2'] - bbox['y1'],
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax.add_patch(rect)

        ax.set_title(f"{instance['category_name']} (ID: {instance['image_id']}, "
                    f"Area: {instance['area']:.0f})")
        ax.axis('off')
        plt.tight_layout()
        plt.show()
