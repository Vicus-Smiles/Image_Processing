#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP2: FILTERS - NOISE REDUCTION AND EDGE DETECTION
Image Processing - Master 1 AI - Unikin

METHODOLOGY:
- Load a base image
- Synthetically generate 3 types of noise: Gaussian, Salt&Pepper, Speckle
- Test 4 filters on each noise type
- Test 2 edge detection methods
- Analyze parameter influence

Author: Okurwoth Vicus ocama
Date: 10/12/2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
IMAGES_DIR = "uploads/TP2"
OUTPUT_DIR = "TP2_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')


class NoiseGenerator:
    """Synthetic noise generator."""

    @staticmethod
    def add_gaussian_noise(image, mean=0, std=25):
        """Add Gaussian noise."""
        noise = np.random.normal(mean, std, image.shape)
        noisy = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
        return noisy

    @staticmethod
    def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
        """Add salt and pepper noise."""
        noisy = image.copy().astype(float)

        # Salt (white)
        salt_mask = np.random.random(image.shape) < salt_prob
        noisy[salt_mask] = 255

        # Pepper (black)
        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy[pepper_mask] = 0

        return noisy.astype(np.uint8)

    @staticmethod
    def add_speckle_noise(image, variance=0.01):
        """Add speckle noise (multiplicative)."""
        noise = np.random.normal(1, variance, image.shape)
        noisy = np.clip(image.astype(float) * noise, 0, 255).astype(np.uint8)
        return noisy


class NoiseReductionAndEdgeDetection:
    """Class for noise reduction and edge detection."""

    def __init__(self, image_path):
        """Initialize with an image."""
        self.image_path = image_path
        self.img = cv2.imread(image_path, 0)  # Grayscale

        if self.img is None:
            raise ValueError(f"Unable to load image: {image_path}")

        self.original = self.img.copy()
        self.height, self.width = self.img.shape
        print(f"✓ Image loaded: {image_path} ({self.width}x{self.height})")

    # ===== NOISE REDUCTION =====

    def mean_filter(self, kernel_size=5):
        """Mean filter."""
        return cv2.blur(self.img, (kernel_size, kernel_size))

    def gaussian_filter(self, kernel_size=5, sigma=1.0):
        """Gaussian filter."""
        return cv2.GaussianBlur(self.img, (kernel_size, kernel_size), sigma)

    def median_filter(self, kernel_size=5):
        """Median filter."""
        return cv2.medianBlur(self.img, kernel_size)

    def morphological_filter(self, kernel_size=5, operation='open'):
        """Morphological filters."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))

        if operation == 'open':
            return cv2.morphologyEx(self.img, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            return cv2.morphologyEx(self.img, cv2.MORPH_CLOSE, kernel)
        elif operation == 'dilate':
            return cv2.dilate(self.img, kernel, iterations=1)
        elif operation == 'erode':
            return cv2.erode(self.img, kernel, iterations=1)

    # ===== EDGE DETECTION =====

    def gradient_threshold(self, threshold=100):
        """Gradient + Threshold detection."""
        sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)

        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        magnitude_normalized = (magnitude / magnitude.max() * 255).astype(np.uint8)

        edges = (magnitude_normalized > threshold).astype(np.uint8) * 255

        return edges, magnitude_normalized

    def canny_edge_detection(self, threshold1=100, threshold2=200):
        """Canny filter detection."""
        blurred = cv2.GaussianBlur(self.img, (5, 5), 1.5)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        return edges

    # ===== METRICS =====

    def calculate_mse(self, img1, img2):
        """Calculate mean squared error."""
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    def calculate_psnr(self, img1, img2):
        """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
        mse = self.calculate_mse(img1, img2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr


def create_noise_comparison_figure(original, noisy_images, noise_names, title):
    """Create a noise comparison figure."""
    n_noises = len(noisy_images)
    fig, axes = plt.subplots(2, n_noises + 1, figsize=(4 * (n_noises + 1), 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Row 1: Noisy images
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    for idx, (noisy, name) in enumerate(zip(noisy_images, noise_names)):
        axes[0, idx + 1].imshow(noisy, cmap='gray')
        axes[0, idx + 1].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[0, idx + 1].axis('off')

    # Row 2: Histograms
    axes[1, 0].hist(original.flatten(), bins=256, color='black', alpha=0.7)
    axes[1, 0].set_title('Hist. Original', fontsize=10)
    axes[1, 0].set_xlim([0, 256])

    for idx, (noisy, name) in enumerate(zip(noisy_images, noise_names)):
        axes[1, idx + 1].hist(noisy.flatten(), bins=256, color='blue', alpha=0.7)
        axes[1, idx + 1].set_title(f'Hist. {name}', fontsize=10)
        axes[1, idx + 1].set_xlim([0, 256])

    plt.tight_layout()
    return fig


def create_filter_comparison_figure(original, noisy, filtered_results, filter_names, title):
    """Create a filter comparison figure."""
    n_filters = len(filtered_results)
    fig, axes = plt.subplots(2, n_filters + 2, figsize=(4 * (n_filters + 2), 8))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Column 1: Original and Noisy
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(noisy, cmap='gray')
    axes[1, 0].set_title('Noisy', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    # Column 2: Difference
    diff = cv2.absdiff(original, noisy)
    axes[0, 1].imshow(diff, cmap='hot')
    axes[0, 1].set_title('Noise Diff.', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 1].text(0.5, 0.5, 'Filters\nApplied', ha='center', va='center',
                    fontsize=12, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')

    # Columns 3+: Filter results
    for idx, (filtered, name) in enumerate(zip(filtered_results, filter_names)):
        col = idx + 2

        # Filtered image
        axes[0, col].imshow(filtered, cmap='gray')
        axes[0, col].set_title(name, fontsize=11, fontweight='bold')
        axes[0, col].axis('off')

        # Difference with original
        diff_filtered = cv2.absdiff(original, filtered)
        axes[1, col].imshow(diff_filtered, cmap='hot')
        axes[1, col].set_title(f'Diff. {name}', fontsize=10)
        axes[1, col].axis('off')

    plt.tight_layout()
    return fig


def create_edge_detection_figure(original, noisy, edge_results, edge_names, title):
    """Create an edge detection comparison figure."""
    n_methods = len(edge_results)
    fig, axes = plt.subplots(1, n_methods + 2, figsize=(4 * (n_methods + 2), 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Noisy
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title('Noisy', fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Results
    for idx, (edges, name) in enumerate(zip(edge_results, edge_names)):
        axes[idx + 2].imshow(edges, cmap='gray')
        axes[idx + 2].set_title(name, fontsize=11, fontweight='bold')
        axes[idx + 2].axis('off')

    plt.tight_layout()
    return fig


def main():
    """Main function."""

    print("\n" + "=" * 80)
    print("TP2: FILTERS - NOISE REDUCTION AND EDGE DETECTION")
    print("=" * 80 + "\n")

    # Find images
    image_files = sorted(Path(IMAGES_DIR).glob("tp2*.png"))
    if not image_files:
        print(f"❌ No images found in {IMAGES_DIR}")
        return

    print(f"📁 Images found: {len(image_files)}")
    for img_file in image_files:
        print(f"   - {img_file.name}")

    # Process first image
    image_path = image_files[0]
    print(f"\n{'=' * 80}")
    print(f"Processing: {image_path.name}")
    print(f"{'=' * 80}\n")

    try:
        nred = NoiseReductionAndEdgeDetection(str(image_path))
        image_name = image_path.stem

        # ===== PART 1: NOISE GENERATION =====
        print("PART 1: SYNTHETIC NOISE GENERATION")
        print("─" * 80)

        print("\n1️⃣  Generating noise...")

        # Generate noise
        gaussian_noisy = NoiseGenerator.add_gaussian_noise(nred.original, std=30)
        salt_pepper_noisy = NoiseGenerator.add_salt_pepper_noise(nred.original,
                                                                 salt_prob=0.02,
                                                                 pepper_prob=0.02)
        speckle_noisy = NoiseGenerator.add_speckle_noise(nred.original, variance=0.05)

        print("   ✓ Gaussian noise generated")
        print("   ✓ Salt & Pepper noise generated")
        print("   ✓ Speckle noise generated")

        # Visualize noise
        fig = create_noise_comparison_figure(
            nred.original,
            [gaussian_noisy, salt_pepper_noisy, speckle_noisy],
            ['Gaussian (σ=30)', 'Salt & Pepper (2%)', 'Speckle (σ=0.05)'],
            'Noise Type Comparison'
        )
        fig.savefig(f"{OUTPUT_DIR}/01_noise_types.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("\n   📊 Figure saved: 01_noise_types.png")

        # ===== PART 2: NOISE REDUCTION =====
        print("\n" + "─" * 80)
        print("PART 2: NOISE REDUCTION - COMPARATIVE ANALYSIS")
        print("─" * 80)

        # For each noise type
        noise_types = [
            ("Gaussian", gaussian_noisy),
            ("Salt & Pepper", salt_pepper_noisy),
            ("Speckle", speckle_noisy)
        ]

        for noise_name, noisy_img in noise_types:
            print(f"\n2️⃣  Processing {noise_name} noise...")

            # Apply filters
            nred.img = noisy_img  # Update image

            mean_filtered = nred.mean_filter(kernel_size=5)
            gaussian_filtered = nred.gaussian_filter(kernel_size=5, sigma=1.0)
            median_filtered = nred.median_filter(kernel_size=5)
            morph_filtered = nred.morphological_filter(kernel_size=5, operation='open')

            # Create figure
            fig = create_filter_comparison_figure(
                nred.original,
                noisy_img,
                [mean_filtered, gaussian_filtered, median_filtered, morph_filtered],
                ['Mean\n(5×5)', 'Gaussian\n(5×5, σ=1)', 'Median\n(5×5)', 'Opening\n(5×5)'],
                f'Noise Reduction: {noise_name}'
            )

            filename = f"02_noise_reduction_{noise_name.replace(' ', '_').lower()}.png"
            fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"   📊 Figure saved: {filename}")

        # ===== PART 3: PARAMETER INFLUENCE =====
        print("\n" + "─" * 80)
        print("PART 3: PARAMETER INFLUENCE")
        print("─" * 80)

        print("\n3️⃣  Kernel size influence (Median Filter)...")

        nred.img = salt_pepper_noisy  # Use salt & pepper noise

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Kernel Size Influence - Median Filter (Salt & Pepper Noise)',
                     fontsize=14, fontweight='bold')

        kernel_sizes = [3, 5, 7, 9, 11, 13]
        for idx, k in enumerate(kernel_sizes):
            ax = axes[idx // 3, idx % 3]
            filtered = nred.median_filter(kernel_size=k)
            psnr = nred.calculate_psnr(nred.original, filtered)

            ax.imshow(filtered, cmap='gray')
            ax.set_title(f'Kernel {k}×{k}\nPSNR={psnr:.2f} dB',
                         fontsize=11, fontweight='bold')
            ax.axis('off')

        plt.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/03_kernel_size_influence.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   📊 Figure saved: 03_kernel_size_influence.png")

        # ===== PART 4: EDGE DETECTION =====
        print("\n" + "─" * 80)
        print("PART 4: EDGE DETECTION")
        print("─" * 80)

        print("\n4️⃣  Edge detection on original image...")

        nred.img = nred.original

        # Gradient + Threshold
        edges_grad_100, _ = nred.gradient_threshold(threshold=100)
        edges_grad_150, _ = nred.gradient_threshold(threshold=150)

        # Canny
        edges_canny_50_150 = nred.canny_edge_detection(50, 150)
        edges_canny_100_200 = nred.canny_edge_detection(100, 200)

        fig = create_edge_detection_figure(
            nred.original,
            nred.original,
            [edges_grad_100, edges_grad_150, edges_canny_50_150, edges_canny_100_200],
            ['Gradient\n(T=100)', 'Gradient\n(T=150)', 'Canny\n(50,150)', 'Canny\n(100,200)'],
            'Edge Detection - Original Image'
        )
        fig.savefig(f"{OUTPUT_DIR}/04_edge_detection_original.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   📊 Figure saved: 04_edge_detection_original.png")

        print("\n5️⃣  Edge detection on noisy image (Salt & Pepper)...")

        nred.img = salt_pepper_noisy

        # Gradient + Threshold
        edges_grad_100_noisy, _ = nred.gradient_threshold(threshold=100)
        edges_grad_150_noisy, _ = nred.gradient_threshold(threshold=150)

        # Canny
        edges_canny_50_150_noisy = nred.canny_edge_detection(50, 150)
        edges_canny_100_200_noisy = nred.canny_edge_detection(100, 200)

        fig = create_edge_detection_figure(
            nred.original,
            salt_pepper_noisy,
            [edges_grad_100_noisy, edges_grad_150_noisy, edges_canny_50_150_noisy, edges_canny_100_200_noisy],
            ['Gradient\n(T=100)', 'Gradient\n(T=150)', 'Canny\n(50,150)', 'Canny\n(100,200)'],
            'Edge Detection - Noisy Image (Salt & Pepper)'
        )
        fig.savefig(f"{OUTPUT_DIR}/05_edge_detection_noisy.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   📊 Figure saved: 05_edge_detection_noisy.png")

        # ===== PART 5: FINAL COMPARISON =====
        print("\n" + "─" * 80)
        print("PART 5: FINAL COMPARISON - BEST APPROACH")
        print("─" * 80)

        print("\n6️⃣  Final comparison...")

        nred.img = salt_pepper_noisy

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Optimal Approach: Filtering + Edge Detection',
                     fontsize=14, fontweight='bold')

        # Row 1: Filtering
        axes[0, 0].imshow(nred.original, cmap='gray')
        axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(salt_pepper_noisy, cmap='gray')
        axes[0, 1].set_title('Noisy (Salt & Pepper)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')

        median_5 = nred.median_filter(kernel_size=5)
        axes[0, 2].imshow(median_5, cmap='gray')
        axes[0, 2].set_title('After Median Filter (5×5)', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')

        # Row 2: Edge detection
        nred.img = nred.original
        edges_original = nred.canny_edge_detection(100, 200)
        axes[1, 0].imshow(edges_original, cmap='gray')
        axes[1, 0].set_title('Canny on Original', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')

        nred.img = salt_pepper_noisy
        edges_noisy = nred.canny_edge_detection(100, 200)
        axes[1, 1].imshow(edges_noisy, cmap='gray')
        axes[1, 1].set_title('Canny on Noisy', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')

        nred.img = median_5
        edges_filtered = nred.canny_edge_detection(100, 200)
        axes[1, 2].imshow(edges_filtered, cmap='gray')
        axes[1, 2].set_title('Canny on Filtered', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')

        plt.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/06_final_comparison.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("   📊 Figure saved: 06_final_comparison.png")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ All results have been saved in:", OUTPUT_DIR)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()