import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class ImageNavigator:
    def __init__(self, json_file_path, example_index=0):
        """Initialize the navigator with JSON data."""
        with open(json_file_path, 'r') as f:
            self.data = json.load(f)

        if example_index >= len(self.data['examples']):
            print(
                f"Error: Example index {example_index} out of range. Available examples: {len(self.data['examples'])}")
            self.valid = False
            return

        self.example = self.data['examples'][example_index]
        self.batch_idx = self.example['batch_index']
        self.inputs = np.array(self.example['inputs'])
        self.targets = np.array(self.example['targets'])
        self.logits = np.array(self.example['logits'])

        # Reshape input from flat array to 4D array (num_images, 3, 32, 32)
        num_images = self.inputs.shape[0] if len(
            self.inputs.shape) > 1 else self.inputs.size // 3072
        self.inputs_reshaped = self.inputs.reshape(num_images, 3, 32, 32)
        self.predictions = np.argmax(self.logits, axis=1)
        self.num_images = num_images

        # Calculate correctness and confidence for each image
        self.is_correct = self.targets == self.predictions
        self.confidences = self._calculate_confidences()

        # Calculate per-class statistics
        self.class_stats = self._calculate_class_stats()

        # Filtered image indices
        self.filter_mode = 'all'  # 'all', 'correct', 'incorrect'
        self.sort_mode = 'original'  # 'original', 'high_confidence', 'low_confidence'
        self.current_image_index = 0
        self.filtered_indices = list(range(self.num_images))
        self.valid = True

        print(f"Model Accuracy: {self.data['accuracy']:.4f}")
        print(f"Average Loss: {self.data['avg_loss']:.4f}\n")
        print("Keyboard Controls:")
        print("  LEFT/RIGHT arrows or 'p'/'n' - Navigate images")
        print("  'c' - Toggle filter (all/correct/incorrect)")
        print("  'h' - Show highest confidence predictions")
        print("  'l' - Show lowest confidence predictions")
        print("  'o' - Return to original order")
        print("  's' - Show per-class statistics")
        print("  Close window to exit.\n")

    def _calculate_confidences(self):
        """Calculate confidence (softmax probability) for each prediction."""
        confidences = []
        for logits in self.logits:
            exp_logits = np.exp(logits - np.max(logits))
            probabilities = exp_logits / np.sum(exp_logits)
            predicted_class = np.argmax(logits)
            confidences.append(probabilities[predicted_class])
        return np.array(confidences)

    def _calculate_class_stats(self):
        """Calculate per-class accuracy statistics."""
        stats = {}
        for class_idx in range(10):
            class_mask = self.targets == class_idx
            if np.sum(class_mask) > 0:
                class_correct = np.sum(self.is_correct[class_mask])
                class_total = np.sum(class_mask)
                stats[CIFAR10_CLASSES[class_idx]] = {
                    'correct': int(class_correct),
                    'total': int(class_total),
                    'accuracy': float(class_correct / class_total)
                }
        return stats

    def update_filtered_indices(self):
        """Update filtered_indices based on current filter_mode."""
        if self.filter_mode == 'correct':
            self.filtered_indices = np.where(self.is_correct)[0].tolist()
        elif self.filter_mode == 'incorrect':
            self.filtered_indices = np.where(~self.is_correct)[0].tolist()
        else:  # 'all'
            self.filtered_indices = list(range(self.num_images))

        if len(self.filtered_indices) == 0:
            print(f"No images found for filter: {self.filter_mode}")
            self.filtered_indices = list(range(self.num_images))
            return

        self.current_image_index = 0

    def sort_by_confidence(self, reverse=False):
        """Sort filtered indices by confidence."""
        sorted_by_conf = sorted(self.filtered_indices,
                                key=lambda i: self.confidences[i],
                                reverse=reverse)
        self.filtered_indices = sorted_by_conf
        self.current_image_index = 0
        if reverse:
            self.sort_mode = 'high_confidence'
            print("Sorted by HIGHEST confidence")
        else:
            self.sort_mode = 'low_confidence'
            print("Sorted by LOWEST confidence")

    def print_class_stats(self):
        """Print per-class statistics."""
        print("\n" + "="*50)
        print("PER-CLASS STATISTICS")
        print("="*50)
        for class_name in CIFAR10_CLASSES:
            if class_name in self.class_stats:
                stats = self.class_stats[class_name]
                print(
                    f"{class_name:12s}: {stats['correct']:3d}/{stats['total']:3d} correct ({stats['accuracy']:5.1%})")
        print("="*50 + "\n")

    def display_image(self, fig, ax):
        """Display the current image."""
        if not self.valid or len(self.filtered_indices) == 0:
            return

        ax.clear()

        # Get actual image index from filtered list
        actual_idx = self.filtered_indices[self.current_image_index]

        # Get current image
        image_data = self.inputs_reshaped[actual_idx].transpose(
            1, 2, 0)

        # Normalize from normalized range back to 0-255 for display
        # Values are normalized with mean/std, so rescale linearly for display
        image_min = image_data.min()
        image_max = image_data.max()
        image_normalized = (image_data - image_min) / (image_max - image_min)
        image = (image_normalized * 255).astype(np.uint8)
        ax.imshow(image)

        # Title with filter info
        filter_tag = f" [{self.filter_mode.upper()}]" if self.filter_mode != 'all' else ""
        ax.set_title(
            f"Batch {self.batch_idx} - Image {self.current_image_index + 1}/{len(self.filtered_indices)}{filter_tag}")
        ax.axis('off')

        # Get classification info
        expected_class = self.targets[actual_idx]
        predicted_class = self.predictions[actual_idx]
        confidence = self.confidences[actual_idx]
        is_correct = self.is_correct[actual_idx]

        print(
            f"\n[Image {self.current_image_index + 1}/{len(self.filtered_indices)} (actual index: {actual_idx})]")
        print(
            f"Expected Classification: {CIFAR10_CLASSES[expected_class]} (class {expected_class})")
        print(
            f"Predicted Classification: {CIFAR10_CLASSES[predicted_class]} (class {predicted_class})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

        # Add to plot
        match = "✓" if is_correct else "✗"
        info_text = f"Expected: {CIFAR10_CLASSES[expected_class]}\nPredicted: {CIFAR10_CLASSES[predicted_class]} {match}\nConfidence: {confidence:.2%}"
        ax.text(0.5, -0.15, info_text, transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.canvas.draw()

    def on_key(self, event, fig, ax):
        """Handle keyboard navigation."""
        if event.key == 'right' or event.key == 'n':
            self.current_image_index = (
                self.current_image_index + 1) % len(self.filtered_indices)
            self.display_image(fig, ax)
        elif event.key == 'left' or event.key == 'p':
            self.current_image_index = (
                self.current_image_index - 1) % len(self.filtered_indices)
            self.display_image(fig, ax)
        elif event.key == 'c':
            # Toggle filter
            if self.filter_mode == 'all':
                self.filter_mode = 'correct'
                print("Filter: Showing CORRECT predictions only")
            elif self.filter_mode == 'correct':
                self.filter_mode = 'incorrect'
                print("Filter: Showing INCORRECT predictions only")
            else:
                self.filter_mode = 'all'
                print("Filter: Showing ALL predictions")
            self.update_filtered_indices()
            self.sort_mode = 'original'
            self.display_image(fig, ax)
        elif event.key == 'h':
            # High confidence
            self.sort_by_confidence(reverse=True)
            self.display_image(fig, ax)
        elif event.key == 'l':
            # Low confidence
            self.sort_by_confidence(reverse=False)
            self.display_image(fig, ax)
        elif event.key == 'o':
            # Original order
            self.update_filtered_indices()
            self.sort_mode = 'original'
            print("Returned to original order")
            self.display_image(fig, ax)
        elif event.key == 's':
            # Show statistics
            self.print_class_stats()


def display_example(json_file_path, example_index=0):
    """
    Load a JSON file and display example images with navigation.

    Args:
        json_file_path: Path to the JSON file
        example_index: Index of the example to display (default: 0)
    """
    navigator = ImageNavigator(json_file_path, example_index)

    if not navigator.valid:
        return

    # Create figure and display first image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    navigator.display_image(fig, ax)

    # Add keyboard event handling
    fig.canvas.mpl_connect('key_press_event',
                           lambda event: navigator.on_key(event, fig, ax))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    json_path = Path("ReLU/data_relu.json")

    if json_path.exists():
        display_example(str(json_path), example_index=0)
    else:
        print(f"File not found: {json_path}")
