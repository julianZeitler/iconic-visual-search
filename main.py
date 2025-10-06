import numpy as np
import cv2
import json
import os
import argparse
import matplotlib.pyplot as plt
from visual_search import VisualSearchModel


def load_annotations(annotations_path: str):
    """Load annotations from JSON file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)

def process_all_images():
    """Process all images with their annotations and create visualizations."""
    # Load annotations
    annotations_path = "images/annotations.json"
    if not os.path.exists(annotations_path):
        print(f"Error: {annotations_path} not found")
        return

    annotations = load_annotations(annotations_path)

    print(f"Processing {len(annotations)} images...")
    print("=" * 50)

    all_results = []

    for image_name, data in annotations.items():
        print(f"\nProcessing: {image_name}")

        # Load image
        image_path = f"images/{image_name}.jpg"
        if not os.path.exists(image_path):
            print(f"  Warning: {image_path} not found, skipping...")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get target location from bounding box
        if not data['bounding_boxes']:
            print(f"  Warning: No bounding boxes found for {image_name}")
            continue

        bbox = data['bounding_boxes'][0]  # Use first target
        # Calculate target center locally
        target_location = (int((bbox['x1'] + bbox['x2']) / 2), int((bbox['y1'] + bbox['y2']) / 2))

        print(f"  Target location: {target_location}")
        print(f"  Image size: {image.shape}")

        # Initialize model and memorize target
        model = VisualSearchModel()
        model.memorize_target(image, bbox)

        # Perform visual search
        fixations = model.visual_search(image)

        # Calculate performance metrics
        final_fixation = fixations[-1]
        distance_to_target = np.sqrt(
            (final_fixation[0] - target_location[0])**2 +
            (final_fixation[1] - target_location[1])**2
        )

        print(f"  Fixations: {len(fixations)}")
        print(f"  Final distance to target: {distance_to_target:.1f} pixels")

        # Store results
        result = {
            'image_name': image_name,
            'image': image,
            'target_location': target_location,
            'bbox': bbox,
            'fixations': fixations,
            'final_distance': distance_to_target,
            'model': model
        }
        all_results.append(result)

    return all_results


def create_summary_plots(results):
    """Create summary plots for all processed images."""
    if not results:
        print("No results to plot")
        return

    n_images = len(results)
    cols = 3
    rows = (n_images + cols - 1) // cols

    # Create large figure for all images
    _, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if n_images == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, result in enumerate(results):
        ax = axes[i]
        image = result['image']
        fixations = result['fixations']
        target_location = result['target_location']
        bbox = result['bbox']

        # Display image
        ax.imshow(image)

        # Draw bounding box
        from matplotlib.patches import Rectangle
        rect = Rectangle((bbox['x1'], bbox['y1']),
                        bbox['x2'] - bbox['x1'],
                        bbox['y2'] - bbox['y1'],
                        linewidth=2, edgecolor='green',
                        facecolor='none', label='Ground Truth')
        ax.add_patch(rect)

        # Plot fixations
        for j, (x, y) in enumerate(fixations):
            ax.plot(x, y, 'ro', markersize=8, alpha=0.7)
            ax.text(x+5, y+5, f'{j+1}', color='red', fontweight='bold')
            if j > 0:
                prev_x, prev_y = fixations[j-1]
                ax.arrow(prev_x, prev_y, x-prev_x, y-prev_y,
                        head_width=8, head_length=10, fc='red', ec='red', alpha=0.7)

        # Mark target center
        ax.plot(target_location[0], target_location[1], 'g*',
                markersize=15, label='Target Center')

        ax.set_title(f'{result["image_name"]}\nFinal distance: {result["final_distance"]:.1f}px')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.suptitle('Visual Search Results for All Images', y=0.98, fontsize=16)
    plt.show()

    distances = [r['final_distance'] for r in results]
    image_names = [r['image_name'] for r in results]

    # Print summary statistics
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    print(f"Total images processed: {len(results)}")
    print(f"Average final distance: {np.mean(distances):.1f} pixels")
    print(f"Median final distance: {np.median(distances):.1f} pixels")
    print(f"Best performance: {min(distances):.1f} pixels ({image_names[distances.index(min(distances))]})")
    print(f"Worst performance: {max(distances):.1f} pixels ({image_names[distances.index(max(distances))]})")


def process_single_image(image_name: str, show_saliency: bool = False):
    """Process a single image with its annotations."""
    # Load annotations
    annotations_path = "images/annotations.json"
    if not os.path.exists(annotations_path):
        print(f"Error: {annotations_path} not found")
        return None

    annotations = load_annotations(annotations_path)

    if image_name not in annotations:
        print(f"Error: Image '{image_name}' not found in annotations")
        print(f"Available images: {list(annotations.keys())}")
        return None

    data = annotations[image_name]
    print(f"Processing: {image_name}")

    # Load image
    image_path = f"images/{image_name}.jpg"
    if not os.path.exists(image_path):
        print(f"  Error: {image_path} not found")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"  Error: Could not load {image_path}")
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get target location from bounding box
    if not data['bounding_boxes']:
        print(f"  Error: No bounding boxes found for {image_name}")
        return None

    bbox = data['bounding_boxes'][0]  # Use first target
    # Calculate target center locally
    target_location = (int((bbox['x1'] + bbox['x2']) / 2), int((bbox['y1'] + bbox['y2']) / 2))

    print(f"  Target location: {target_location}")
    print(f"  Image size: {image.shape}")

    # Initialize model and memorize target
    model = VisualSearchModel()
    model.memorize_target(image, bbox)

    # Perform visual search
    fixations = model.visual_search(image)

    # Calculate performance metrics
    final_fixation = fixations[-1]
    distance_to_target = np.sqrt(
        (final_fixation[0] - target_location[0])**2 +
        (final_fixation[1] - target_location[1])**2
    )

    print(f"  Fixations: {len(fixations)}")
    for i, (x, y) in enumerate(fixations):
        dist = np.sqrt((x - target_location[0])**2 + (y - target_location[1])**2)
        print(f"    Fixation {i+1}: ({x}, {y}) - Distance: {dist:.1f}px")

    print(f"  Final distance to target: {distance_to_target:.1f} pixels")

    # Visualize results
    print("  Creating basic visualization...")
    model.visualize_search(image, fixations, target_location)
    if show_saliency:
        print("  Creating consecutive saliency maps visualization...")
        model.visualize_consecutive_saliency_maps(image, fixations, target_location)
    
    return {
        'image_name': image_name,
        'image': image,
        'target_location': target_location,
        'bbox': bbox,
        'fixations': fixations,
        'final_distance': distance_to_target,
        'model': model
    }


def process_outside_case(image_name: str = "astronauts", crop_x: int = 350, show_filters: bool = False, show_saliency: bool = False):
    """
    Process the 'outside' case where target is memorized from original image
    but search is performed on a cropped image where target is not present.

    Args:
        image_name: Name of the image to use (default: "astronauts")
        crop_x: X coordinate where to crop the image (default: 350)
        show_filters: Whether to show filter responses
        show_saliency: Whether to show saliency maps
    """
    # Load annotations
    annotations_path = "images/annotations.json"
    if not os.path.exists(annotations_path):
        print(f"Error: {annotations_path} not found")
        return None

    annotations = load_annotations(annotations_path)

    if image_name not in annotations:
        print(f"Error: Image '{image_name}' not found in annotations")
        print(f"Available images: {list(annotations.keys())}")
        return None

    data = annotations[image_name]
    print(f"Processing 'outside' case with: {image_name}")

    # Load original image
    image_path = f"images/{image_name}.jpg"
    if not os.path.exists(image_path):
        print(f"  Error: {image_path} not found")
        return None

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"  Error: Could not load {image_path}")
        return None

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Get target location from bounding box
    if not data['bounding_boxes']:
        print(f"  Error: No bounding boxes found for {image_name}")
        return None

    bbox = data['bounding_boxes'][0]  # Use first target
    target_location = (int((bbox['x1'] + bbox['x2']) / 2), int((bbox['y1'] + bbox['y2']) / 2))

    print(f"  Original image size: {original_image.shape}")
    print(f"  Target location in original: {target_location}")
    print(f"  Cropping at x={crop_x}")

    # Create cropped image (remove everything from x=crop_x onwards)
    cropped_image = original_image[:, :crop_x, :]
    print(f"  Cropped image size: {cropped_image.shape}")

    # Check if target is still in cropped image
    target_in_cropped = target_location[0] < crop_x
    print(f"  Target present in cropped image: {target_in_cropped}")

    # Initialize model and memorize target from ORIGINAL image
    model = VisualSearchModel()
    model.memorize_target(original_image, bbox)
    print("  Target memorized from original image")

    # Perform visual search on CROPPED image
    fixations = model.visual_search(cropped_image)
    print("  Visual search performed on cropped image")

    # Visualize results
    if show_filters:
        print("  Creating detailed visualization with filter responses...")
        # For outside case, don't show target location since it's not in the search image
        model.visualize_search(cropped_image, fixations, None)
    elif show_saliency:
        print("  Creating consecutive saliency maps visualization...")
        model.visualize_consecutive_saliency_maps(cropped_image, fixations, None)
    else:
        print("  Creating basic visualization...")
        model.visualize_search(cropped_image, fixations, None)

    return {
        'image_name': f"{image_name}_outside",
        'original_image': original_image,
        'cropped_image': cropped_image,
        'crop_x': crop_x,
        'target_location': target_location,
        'target_in_cropped': target_in_cropped,
        'bbox': bbox,
        'fixations': fixations,
        'model': model
    }


def create_individual_plots(results):
    """Create detailed individual plots for each image."""
    for result in results:
        print(f"\nCreating detailed plot for: {result['image_name']}")
        result['model'].visualize_search(
            result['image'],
            result['fixations'],
            result['target_location']
        )


def main():
    parser = argparse.ArgumentParser(
        description='Visual Search Model - Process images with iconic visual search',
        epilog='Examples:\n'
               '  python main.py ladybug           # Process ladybug image\n'
               '  python main.py all               # Process all images\n'
               '  python main.py ladybug --saliency # Show consecutive saliency maps\n'
               '  python main.py list              # List available images\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('command', nargs='?',
                       help='Image name, "all", or "list"')
    parser.add_argument('--details', '-d', action='store_true',
                       help='Create individual detailed plots for each image (for "all" mode)')
    parser.add_argument('--saliency', '-s', action='store_true',
                       help='Show consecutive saliency maps for each fixation (for single image mode)')

    args = parser.parse_args()

    command = args.command

    if not command:
        # No command provided
        print("Visual Search Model")
        print("=" * 50)
        print("Usage: python main.py <command> [options]")
        print()
        print("Commands:")
        print("  <image_name>  : Process single image (e.g., 'ladybug', 'monkey')")
        print("  all           : Process all images with summary plots")
        print("  outside       : Process outside case (target memorized from original, search on cropped)")
        print("  list          : List available images")
        print()
        print("Options:")
        print("  --saliency/-s  : Show consecutive saliency maps (for single image)")
        print("  --details/-d   : Create individual detailed plots (for 'all' mode)")
        print()
        print("Examples:")
        print("  python main.py ladybug")
        print("  python main.py ladybug --saliency")
        print("  python main.py outside --saliency")
        print("  python main.py all")
        print("  python main.py all --details")
        print("  python main.py list")
        return

    if command.lower() == 'list':
        # List available images
        annotations_path = "images/annotations.json"
        if os.path.exists(annotations_path):
            annotations = load_annotations(annotations_path)
            print("Available images:")
            for image_name in annotations.keys():
                print(f"  - {image_name}")
        else:
            print(f"Error: {annotations_path} not found")
        return

    if command.lower() == 'outside':
        # Process outside case
        print("Visual Search Model - Processing Outside Case")
        print("=" * 50)

        result = process_outside_case(show_filters=args.filters, show_saliency=args.saliency)
        return

    if command.lower() == 'all':
        # Process all images
        print("Visual Search Model - Processing All Images")
        print("=" * 50)

        results = process_all_images()

        if results:
            # Create summary plots
            create_summary_plots(results)

            # Create individual detailed plots if requested
            if args.details:
                print("\nCreating individual detailed plots...")
                create_individual_plots(results)
            else:
                # Ask user if they want detailed plots
                create_individual = input("\nCreate individual detailed plots for each image? (y/n): ").lower().strip()
                if create_individual == 'y':
                    create_individual_plots(results)

            print("\nProcessing complete!")
        else:
            print("No images were successfully processed.")
        return

    # Assume it's an image name
    image_name = command
    print(f"Visual Search Model - Processing Single Image: {image_name}")
    print("=" * 50)

    result = process_single_image(image_name, show_saliency=args.saliency)
    if result:
        print(f"\nProcessing complete for {image_name}!")
        print(f"Final distance to target: {result['final_distance']:.1f} pixels")


# Example usage
if __name__ == "__main__":
    main()