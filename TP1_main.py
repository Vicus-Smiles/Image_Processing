"""
TP1: CONTRAST CORRECTION - COMPLETE SYSTEM FOR ALL 4 IMAGES
Master 1 - Image Processing
Professional VS Code Implementation

This program processes 4 images with 3 contrast correction methods:
1. Linear Transformation: g(x,y) = a*f(x,y) + c
2. Gamma Correction: g(x,y) = 255 * (f(x,y)/255)^(1/gamma)
3. Histogram Equalization: Automatic

Author: [Your Name]
ID: [Your ID]
Date: [Current Date]
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============================================================
# CONFIGURATION - IMAGE DEFINITIONS
# ============================================================

class Config:
    """Configuration for all 4 images and their optimal parameters"""
    
    IMAGES = {
        1: {
            'name': 'Foggy Forest - Deer',
            'filename': 'image1.jpg',
            'type': 'dark_low_contrast',
            'description': 'Very dark and foggy image with low contrast',
            'linear_params': [
                {'a': 1.2, 'c': 0, 'desc': 'Gentle Contrast'},
                {'a': 1.0, 'c': 30, 'desc': 'Brightness Only'},
                {'a': 1.3, 'c': 20, 'desc': 'Balanced'},
            ],
            'gamma_params': [
                {'gamma': 0.5, 'desc': 'Strong Brightening ⭐'},
                {'gamma': 0.67, 'desc': 'Moderate Brightening'},
                {'gamma': 0.4, 'desc': 'Extreme Brightening'},
            ],
        },
        2: {
            'name': 'Buildings - Urban Scene',
            'filename': 'image2.jpg',
            'type': 'normal_low_contrast',
            'description': 'Medium brightness with low contrast buildings',
            'linear_params': [
                {'a': 1.2, 'c': 0, 'desc': 'Contrast Increase'},
                {'a': 1.0, 'c': 20, 'desc': 'Slight Brightening'},
                {'a': 1.1, 'c': 15, 'desc': 'Balanced'},
            ],
            'gamma_params': [
                {'gamma': 0.8, 'desc': 'Gentle Brightening'},
                {'gamma': 0.9, 'desc': 'Slight Adjustment ⭐'},
                {'gamma': 1.1, 'desc': 'Slight Darkening'},
            ],
        },
        3: {
            'name': 'Snowy Trees - Winter Landscape',
            'filename': 'image3.jpg',
            'type': 'bright_medium_contrast',
            'description': 'Bright snowy scene with medium contrast',
            'linear_params': [
                {'a': 0.9, 'c': 0, 'desc': 'Gentle Compression'},
                {'a': 0.8, 'c': -10, 'desc': 'Compression + Darkening'},
                {'a': 1.0, 'c': 0, 'desc': 'No Change'},
            ],
            'gamma_params': [
                {'gamma': 0.95, 'desc': 'Very Slight Brightening'},
                {'gamma': 1.1, 'desc': 'Slight Darkening ⭐'},
                {'gamma': 1.0, 'desc': 'No Change'},
            ],
        },
        4: {
            'name': 'Dark Room - Interior',
            'filename': 'image4.jpg',
            'type': 'very_dark_extreme',
            'description': 'Extremely dark room with backlit window',
            'linear_params': [
                {'a': 1.5, 'c': 0, 'desc': 'Strong Contrast'},
                {'a': 1.0, 'c': 50, 'desc': 'Strong Brightness'},
                {'a': 2.0, 'c': 30, 'desc': 'Maximum Enhancement'},
            ],
            'gamma_params': [
                {'gamma': 0.4, 'desc': 'Extreme Brightening'},
                {'gamma': 0.5, 'desc': 'Strong Brightening ⭐'},
                {'gamma': 0.33, 'desc': 'Maximum Brightening'},
            ],
        },
    }
    
    # Folder configuration
    INPUT_FOLDER = 'images'
    OUTPUT_FOLDER = 'results'
    REPORT_FOLDER = 'report'
    
    # Figure configuration
    DPI = 150
    FIGURE_SIZE = (14, 24)
    
    # Report configuration
    REPORT_FILENAME = 'TP1_Report.txt'


# ============================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================

class ImageProcessor:
    """Core image processing algorithms"""
    
    @staticmethod
    def linear_transform(img, a, c):
        """
        Linear transformation: g(x,y) = a*f(x,y) + c
        
        Parameters:
            img: Input image (uint8)
            a: Contrast multiplier (a > 1 increases, 0 < a < 1 decreases)
            c: Brightness offset (c > 0 brightens, c < 0 darkens)
        
        Returns:
            Transformed image (uint8), values clipped to [0, 255]
        """
        result = img.astype(np.float32)
        result = a * result + c
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)
    
    @staticmethod
    def gamma_correction(img, gamma):
        """
        Gamma correction: g(x,y) = 255 * (f(x,y)/255)^(1/gamma)
        
        Parameters:
            img: Input image (uint8)
            gamma: Gamma value
                   gamma < 1: brightening (e.g., 0.5)
                   gamma > 1: darkening (e.g., 2.0)
                   gamma = 1: no change
        
        Returns:
            Transformed image (uint8)
        """
        normalized = img.astype(np.float32) / 255.0
        result = np.power(normalized, 1/gamma)
        return (result * 255).astype(np.uint8)
    
    @staticmethod
    def histogram_equalization(img):
        """
        Histogram equalization using OpenCV
        Automatically stretches histogram to use full [0, 255] range
        
        Parameters:
            img: Input image (uint8)
        
        Returns:
            Equalized image (uint8)
        """
        return cv2.equalizeHist(img)
    
    @staticmethod
    def calculate_histogram(img):
        """Calculate image histogram"""
        return cv2.calcHist([img], [0], None, [256], [0, 256])


# ============================================================
# VISUALIZATION
# ============================================================

class Visualizer:
    """Handles visualization and figure generation"""
    
    @staticmethod
    def create_results_figure(image_num, img_original, results):
        """
        Create comprehensive visualization of all results
        
        Parameters:
            image_num: Image number (1-4)
            img_original: Original image array
            results: Dictionary with all processed results
        
        Returns:
            matplotlib figure object
        """
        config = Config.IMAGES[image_num]
        fig, axes = plt.subplots(8, 2, figsize=Config.FIGURE_SIZE)
        
        # ===== ROW 0: ORIGINAL =====
        axes[0, 0].imshow(img_original, cmap='gray')
        axes[0, 0].set_title(f"ORIGINAL - {config['name']}", 
                            fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        hist_orig = ImageProcessor.calculate_histogram(img_original)
        axes[0, 1].plot(hist_orig, color='black', linewidth=2)
        axes[0, 1].set_title('ORIGINAL HISTOGRAM', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Pixel Intensity (0-255)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # ===== ROWS 1-3: LINEAR TRANSFORMATIONS =====
        for row, result in enumerate(list(results['linear'].values())[:3], 1):
            params = result['params']
            axes[row, 0].imshow(result['image'], cmap='gray')
            axes[row, 0].set_title(
                f"Linear: a={params['a']}, c={params['c']}\n({params['desc']})",
                fontsize=11
            )
            axes[row, 0].axis('off')
            
            hist = ImageProcessor.calculate_histogram(result['image'])
            axes[row, 1].plot(hist, color='blue', linewidth=2)
            axes[row, 1].set_title(f"Histogram: a={params['a']}, c={params['c']}", 
                                  fontsize=11)
            axes[row, 1].set_xlabel('Pixel Intensity (0-255)')
            axes[row, 1].set_ylabel('Frequency')
            axes[row, 1].grid(True, alpha=0.3)
        
        # ===== ROWS 4-6: GAMMA CORRECTIONS =====
        for row, result in enumerate(list(results['gamma'].values())[:3], 4):
            params = result['params']
            is_best = '⭐' in params['desc']
            color = 'green' if is_best else 'black'
            weight = 'bold' if is_best else 'normal'
            
            axes[row, 0].imshow(result['image'], cmap='gray')
            axes[row, 0].set_title(
                f"Gamma: gamma={params['gamma']}\n({params['desc']})",
                fontsize=11, color=color, fontweight=weight
            )
            axes[row, 0].axis('off')
            
            hist = ImageProcessor.calculate_histogram(result['image'])
            axes[row, 1].plot(hist, color='green', linewidth=2)
            axes[row, 1].set_title(f"Histogram: gamma={params['gamma']}", 
                                  fontsize=11, color=color, fontweight=weight)
            axes[row, 1].set_xlabel('Pixel Intensity (0-255)')
            axes[row, 1].set_ylabel('Frequency')
            axes[row, 1].grid(True, alpha=0.3)
        
        # ===== ROW 7: HISTOGRAM EQUALIZATION =====
        axes[7, 0].imshow(results['equalization']['image'], cmap='gray')
        axes[7, 0].set_title('⭐ Histogram Equalization (Automatic)',
                            fontsize=11, color='red', fontweight='bold')
        axes[7, 0].axis('off')
        
        hist_eq = ImageProcessor.calculate_histogram(
            results['equalization']['image'])
        axes[7, 1].plot(hist_eq, color='red', linewidth=2)
        axes[7, 1].set_title('⭐ Equalization Histogram', 
                            fontsize=11, color='red', fontweight='bold')
        axes[7, 1].set_xlabel('Pixel Intensity (0-255)')
        axes[7, 1].set_ylabel('Frequency')
        axes[7, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, image_num, output_folder):
        """Save figure to PNG file"""
        filename = f'image{image_num}_results.png'
        filepath = os.path.join(output_folder, filename)
        fig.savefig(filepath, dpi=Config.DPI, bbox_inches='tight')
        plt.close(fig)
        return filepath


# ============================================================
# MAIN PROCESSOR
# ============================================================

class TP1Processor:
    """Main processor for TP1 assignment"""
    
    def __init__(self):
        """Initialize processor and create folders"""
        self.setup_folders()
        self.all_results = {}
    
    def setup_folders(self):
        """Create necessary directories"""
        os.makedirs(Config.INPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(Config.REPORT_FOLDER, exist_ok=True)
        
        print("="*70)
        print("TP1: CONTRAST CORRECTION - SYSTEM INITIALIZED")
        print("="*70)
        print(f"\n✓ Folders ready:")
        print(f"  Input:  {Config.INPUT_FOLDER}/")
        print(f"  Output: {Config.OUTPUT_FOLDER}/")
        print(f"  Report: {Config.REPORT_FOLDER}/")
    
    def load_image(self, image_num):
        """Load a single image"""
        config = Config.IMAGES[image_num]
        filepath = os.path.join(Config.INPUT_FOLDER, config['filename'])
        
        print(f"\n[IMAGE {image_num}] Loading: {config['name']}")
        print(f"  File: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"  ❌ ERROR: File not found!")
            print(f"     Make sure '{config['filename']}' is in '{Config.INPUT_FOLDER}/' folder")
            return None
        
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"  ❌ ERROR: Could not load image!")
            return None
        
        print(f"  ✓ Loaded successfully")
        print(f"    Dimensions: {image.shape}")
        print(f"    Pixel range: [{image.min()}, {image.max()}]")
        
        return image
    
    def process_image(self, image_num, img):
        """Process single image with all methods"""
        config = Config.IMAGES[image_num]
        print(f"\n[IMAGE {image_num}] Processing...")
        
        results = {
            'linear': {},
            'gamma': {},
            'equalization': {}
        }
        
        # Linear transformations
        print(f"  Applying linear transformations...")
        for idx, params in enumerate(config['linear_params'], 1):
            result_img = ImageProcessor.linear_transform(img, params['a'], params['c'])
            key = f"linear_{idx}"
            results['linear'][key] = {
                'image': result_img,
                'params': params
            }
            print(f"    ✓ a={params['a']}, c={params['c']}: {params['desc']}")
        
        # Gamma corrections
        print(f"  Applying gamma corrections...")
        for idx, params in enumerate(config['gamma_params'], 1):
            result_img = ImageProcessor.gamma_correction(img, params['gamma'])
            key = f"gamma_{idx}"
            results['gamma'][key] = {
                'image': result_img,
                'params': params
            }
            print(f"    ✓ gamma={params['gamma']}: {params['desc']}")
        
        # Histogram equalization
        print(f"  Applying histogram equalization...")
        eq_img = ImageProcessor.histogram_equalization(img)
        results['equalization'] = {
            'image': eq_img,
            'params': {'method': 'cv2.equalizeHist()'}
        }
        print(f"    ✓ Automatic equalization applied")
        
        return results
    
    def generate_report(self):
        """Generate refined report focusing on explanation and results analysis"""
        report_path = os.path.join(Config.REPORT_FOLDER, Config.REPORT_FILENAME)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("TP1: CONTRAST CORRECTION\n")
            f.write("="*80 + "\n\n")
            
            # Student Info
            f.write("STUDENT INFORMATION:\n")
            f.write("-"*80 + "\n")
            f.write("Name: [Your Name]\n")
            f.write("ID: [Your ID]\n")
            f.write(f"Date: {datetime.now().strftime('%B %d, %Y')}\n\n")
            
            # Part 1: Brief Explanation
            f.write("="*80 + "\n")
            f.write("1. BRIEF EXPLANATION OF WHAT YOU DID\n")
            f.write("="*80 + "\n\n")
            
            f.write("This work implements and tests three contrast correction methods:\n\n")
            
            f.write("METHOD 1: LINEAR TRANSFORMATION\n")
            f.write("─────────────────────────────────\n")
            f.write("Formula: g(x,y) = a·f(x,y) + c\n\n")
            f.write("Where:\n")
            f.write("  • f(x,y) = input pixel intensity at position (x,y)\n")
            f.write("  • g(x,y) = output enhanced pixel\n")
            f.write("  • a = contrast multiplier\n")
            f.write("    - a > 1: increases contrast (stretches pixel values)\n")
            f.write("    - 0 < a < 1: decreases contrast (compresses pixel values)\n")
            f.write("  • c = brightness offset\n")
            f.write("    - c > 0: increases brightness (shifts histogram right)\n")
            f.write("    - c < 0: decreases brightness (shifts histogram left)\n\n")
            f.write("Implementation: Custom implementation in Python using NumPy arrays\n")
            f.write("Values clipped to [0, 255] range to prevent overflow\n\n")
            
            f.write("METHOD 2: GAMMA CORRECTION\n")
            f.write("──────────────────────────\n")
            f.write("Formula: g(x,y) = 255 × [f(x,y)/255]^(1/γ)\n\n")
            f.write("Where:\n")
            f.write("  • γ (gamma) is the correction parameter\n")
            f.write("  • γ < 1: brightening transformation (e.g., 0.5)\n")
            f.write("    - Non-linear brightening\n")
            f.write("    - Dark areas brighten more than bright areas\n")
            f.write("  • γ > 1: darkening transformation (e.g., 2.0)\n")
            f.write("    - Non-linear darkening\n")
            f.write("    - Bright areas darken more than dark areas\n")
            f.write("  • γ = 1: no transformation\n\n")
            f.write("Implementation: Custom implementation using power function\n")
            f.write("No clipping needed (stays naturally in [0, 255] range)\n")
            f.write("Produces more natural-looking results than linear transformation\n\n")
            
            f.write("METHOD 3: HISTOGRAM EQUALIZATION\n")
            f.write("────────────────────────────────\n")
            f.write("Formula: g(x,y) = (L-1) × CDF[f(x,y)]\n\n")
            f.write("Where:\n")
            f.write("  • CDF = Cumulative Distribution Function of the histogram\n")
            f.write("  • L = number of intensity levels (256 for 8-bit image)\n")
            f.write("  • CDF transforms pixel values to spread across full range\n\n")
            f.write("Implementation: OpenCV function cv2.equalizeHist()\n")
            f.write("Automatic method: no parameters to adjust\n")
            f.write("Stretches histogram to use maximum dynamic range [0, 255]\n\n")
            
            f.write("TESTING APPROACH:\n")
            f.write("─────────────────\n")
            f.write("For each of the 4 images:\n")
            f.write("  • 3 linear transformations with different (a, c) values\n")
            f.write("  • 3 gamma corrections with different γ values\n")
            f.write("  • 1 histogram equalization (automatic)\n")
            f.write("  • Total: 7 results per image to show parameter influence\n\n")
            f.write("Purpose: Demonstrate how different parameter values affect image\n")
            f.write("quality and histogram distribution\n\n")
            
            # Part 2: Visualization of Results
            f.write("="*80 + "\n")
            f.write("2. VISUALIZATION OF THE RESULTS OBTAINED\n")
            f.write("="*80 + "\n\n")
            
            f.write("Each result PNG file contains:\n")
            f.write("  • Left column: Enhanced image\n")
            f.write("  • Right column: Histogram of that image\n")
            f.write("  • Row 0: Original image + original histogram (reference)\n")
            f.write("  • Rows 1-3: Linear transformations with parameter values\n")
            f.write("  • Rows 4-6: Gamma corrections with parameter values\n")
            f.write("  • Row 7: Histogram equalization (automatic)\n\n")
            
            # Detailed results for each image
            for num, config in Config.IMAGES.items():
                f.write("\n" + "-"*80 + "\n")
                f.write(f"IMAGE {num}: {config['name']}\n")
                f.write("-"*80 + "\n\n")
                
                f.write(f"File: {config['filename']}\n")
                f.write(f"Type: {config['type']}\n")
                f.write(f"Description: {config['description']}\n")
                f.write(f"Output file: image{num}_results.png\n\n")
                
                f.write("RESULTS OBTAINED:\n\n")
                
                # Linear transformations
                f.write("LINEAR TRANSFORMATIONS:\n")
                for idx, params in enumerate(config['linear_params'], 1):
                    f.write(f"\n  [{idx}] a={params['a']}, c={params['c']} ({params['desc']})\n")
                    f.write(f"      Formula: g(x,y) = {params['a']}·f(x,y) + {params['c']}\n")
                
                # Gamma corrections
                f.write("\n\nGAMMA CORRECTIONS:\n")
                for idx, params in enumerate(config['gamma_params'], 1):
                    f.write(f"\n  [{idx}] γ={params['gamma']} ({params['desc']})\n")
                    f.write(f"      Formula: g(x,y) = 255 × [f(x,y)/255]^(1/{params['gamma']})\n")
                
                # Equalization
                f.write("\n\nHISTOGRAM EQUALIZATION:\n")
                f.write("\n  [Auto] Automatic (no parameters)\n")
                f.write("         Formula: g(x,y) = 255 × CDF[f(x,y)]\n\n")
            
            # Part 3: Feedback on Results
            f.write("\n" + "="*80 + "\n")
            f.write("3. FEEDBACK ON RESULTS OBTAINED\n")
            f.write("="*80 + "\n\n")
            
            f.write("EFFECTS OF METHODS ON IMAGES:\n")
            f.write("-"*80 + "\n\n")
            
            f.write("IMAGE 1 - FOGGY FOREST (Dark, Low Contrast):\n\n")
            f.write("Original histogram characteristics:\n")
            f.write("  • Concentrated in dark region (left side)\n")
            f.write("  • Peak around 50-100 intensity values\n")
            f.write("  • Very low contrast\n\n")
            f.write("Linear Transformation effects:\n")
            f.write("  • a=1.2, c=0: Histogram stretches, some clipping at 255\n")
            f.write("  • a=1.0, c=30: Shifts histogram right, maintains shape\n")
            f.write("  • a=1.3, c=20: Strong stretch + shift, more clipping\n\n")
            f.write("Gamma Correction effects:\n")
            f.write("  • γ=0.5: Strong brightening, histogram spreads nicely\n")
            f.write("  • γ=0.67: Moderate brightening, good balance\n")
            f.write("  • γ=0.4: Extreme brightening, very bright image\n\n")
            f.write("Histogram Equalization: Stretches to full [0,255] range\n\n")
            
            f.write("IMAGE 2 - BUILDINGS (Normal, Low Contrast):\n\n")
            f.write("Original histogram characteristics:\n")
            f.write("  • Medium concentration in mid-range\n")
            f.write("  • Moderate contrast\n")
            f.write("  • Some pixels across full range\n\n")
            f.write("Linear Transformation effects:\n")
            f.write("  • a=1.2, c=0: Increases contrast, histogram spreads\n")
            f.write("  • a=1.0, c=20: Slight brightness increase\n")
            f.write("  • a=1.1, c=15: Balanced enhancement\n\n")
            f.write("Gamma Correction effects:\n")
            f.write("  • γ=0.8: Slight brightening\n")
            f.write("  • γ=0.9: Very subtle adjustment\n")
            f.write("  • γ=1.1: Slight darkening\n\n")
            f.write("Histogram Equalization: Good improvement for low-contrast scene\n\n")
            
            f.write("IMAGE 3 - SNOWY TREES (Bright, Medium Contrast):\n\n")
            f.write("Original histogram characteristics:\n")
            f.write("  • Concentrated in bright region (right side)\n")
            f.write("  • Already has decent contrast\n")
            f.write("  • Washed-out appearance\n\n")
            f.write("Linear Transformation effects:\n")
            f.write("  • a=0.9, c=0: Slight compression, minimal change\n")
            f.write("  • a=0.8, c=-10: More compression, darkens image\n")
            f.write("  • a=1.0, c=0: No change (reference)\n\n")
            f.write("Gamma Correction effects:\n")
            f.write("  • γ=0.95: Minimal brightening\n")
            f.write("  • γ=1.1: Slight darkening, improves appearance\n")
            f.write("  • γ=1.0: No change (reference)\n\n")
            f.write("Histogram Equalization: Limited benefit (already spread out)\n\n")
            
            f.write("IMAGE 4 - DARK ROOM (Extreme Dark, Backlit):\n\n")
            f.write("Original histogram characteristics:\n")
            f.write("  • Almost entirely in dark region\n")
            f.write("  • Only small spike from backlit window\n")
            f.write("  • Extreme low contrast\n")
            f.write("  • Most pixels nearly black (0-50)\n\n")
            f.write("Linear Transformation effects:\n")
            f.write("  • a=1.5, c=0: Strong stretch, significant brightening\n")
            f.write("  • a=1.0, c=50: Adds brightness uniformly\n")
            f.write("  • a=2.0, c=30: Maximum enhancement, reveals window detail\n\n")
            f.write("Gamma Correction effects:\n")
            f.write("  • γ=0.4: Extreme brightening, very effective\n")
            f.write("  • γ=0.5: Strong brightening, good balance\n")
            f.write("  • γ=0.33: Maximum brightening, window very prominent\n\n")
            f.write("Histogram Equalization: Very effective, stretches entire range\n\n")
            
            f.write("\nINFLUENCE OF PARAMETERS:\n")
            f.write("-"*80 + "\n\n")
            
            f.write("LINEAR TRANSFORMATION PARAMETER INFLUENCE:\n\n")
            f.write("Parameter 'a' (Contrast multiplier):\n")
            f.write("  • INCREASING a: Histogram stretches (wider range)\n")
            f.write("    - a=2.0: Stronger stretching, more clipping at edges\n")
            f.write("    - a=1.5: Moderate stretching\n")
            f.write("    - a=1.2: Gentle stretching\n")
            f.write("    - a<1: Compression (opposite effect)\n")
            f.write("  • Visible effect: Increased contrast in image\n\n")
            f.write("Parameter 'c' (Brightness offset):\n")
            f.write("  • INCREASING c: Histogram shifts right (brighter)\n")
            f.write("    - c=50: Large rightward shift\n")
            f.write("    - c=30: Moderate shift\n")
            f.write("    - c=20: Gentle shift\n")
            f.write("    - c<0: Leftward shift (darker)\n")
            f.write("  • Visible effect: Overall brightness increase\n")
            f.write("  • Warning: Large c values cause clipping at 255\n\n")
            
            f.write("GAMMA CORRECTION PARAMETER INFLUENCE:\n\n")
            f.write("Parameter 'γ' (Gamma value):\n")
            f.write("  • DECREASING γ (γ<1): Non-linear brightening\n")
            f.write("    - γ=0.4: Extreme brightening (largest effect)\n")
            f.write("    - γ=0.5: Strong brightening\n")
            f.write("    - γ=0.67: Moderate brightening\n")
            f.write("    - γ=0.8: Gentle brightening\n")
            f.write("  • INCREASING γ (γ>1): Non-linear darkening\n")
            f.write("    - γ=1.1: Gentle darkening\n")
            f.write("    - γ=1.5: Moderate darkening\n")
            f.write("    - γ=2.0: Strong darkening\n")
            f.write("    - γ=2.5: Extreme darkening\n")
            f.write("  • Visible effect: Natural-looking brightness change\n")
            f.write("  • Advantage: NO clipping at edges\n")
            f.write("  • Key feature: Non-linear (dark areas change more)\n\n")
            
            f.write("HISTOGRAM EQUALIZATION CHARACTERISTICS:\n\n")
            f.write("  • No parameters: Fully automatic\n")
            f.write("  • Effect: Spreads histogram to use entire [0,255] range\n")
            f.write("  • CDF-based: Uses cumulative distribution function\n")
            f.write("  • Best for: Extremely dark or bright images\n")
            f.write("  • Limitation: Can amplify noise in uniform regions\n\n")
            
            f.write("\nCOMPARATIVE ANALYSIS:\n")
            f.write("-"*80 + "\n\n")
            
            f.write("LINEAR vs GAMMA:\n")
            f.write("  • Linear: Direct multiplication, simple but can clip\n")
            f.write("  • Gamma: Non-linear, no clipping, more natural appearance\n")
            f.write("  • Best choice: Gamma usually produces better visual results\n\n")
            
            f.write("ALL METHODS vs HISTOGRAM EQUALIZATION:\n")
            f.write("  • Manual methods: Precise control via parameters\n")
            f.write("  • Equalization: Automatic, optimal histogram spread\n")
            f.write("  • Equalization wins: For maximum contrast recovery\n")
            f.write("  • Manual wins: For preserving specific image characteristics\n\n")
            
            f.write("OPTIMAL PARAMETER SELECTIONS BY IMAGE TYPE:\n")
            f.write("  • Very dark images: γ<1 (e.g., 0.4-0.5) or Equalization\n")
            f.write("  • Dark images: γ<1 (e.g., 0.5-0.67) or Linear a>1\n")
            f.write("  • Normal images: γ≈1 or a≈1.2, small adjustments\n")
            f.write("  • Bright images: γ>1 (e.g., 1.5-2.0) or Linear a<1\n")
            f.write("  • Very bright images: γ>1 (e.g., 2.0-2.5) or Equalization\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Report saved: {report_path}") 
        
        print(f"\n✓ Report saved: {report_path}")
    
    def process_all(self):
        """Process all 4 images"""
        print("\n" + "="*70)
        print("STARTING FULL PIPELINE")
        print("="*70)
        
        for image_num in range(1, 5):
            # Load image
            img = self.load_image(image_num)
            if img is None:
                print(f"⚠ Skipping Image {image_num}\n")
                continue
            
            # Process image
            results = self.process_image(image_num, img)
            self.all_results[image_num] = results
            
            # Visualize and save
            print(f"  Generating visualization...")
            fig = Visualizer.create_results_figure(image_num, img, results)
            filepath = Visualizer.save_figure(fig, image_num, Config.OUTPUT_FOLDER)
            print(f"  ✓ Saved: {filepath}")
        
        # Generate report
        print(f"\nGenerating report...")
        self.generate_report()
        
        print("\n" + "="*70)
        print("✓ COMPLETE! All images processed successfully")
        print("="*70)
        print(f"\n📁 Results location:")
        print(f"   {os.path.abspath(Config.OUTPUT_FOLDER)}/")
        print(f"\n📄 Report location:")
        print(f"   {os.path.abspath(Config.REPORT_FOLDER)}/")
        print("\n" + "="*70)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        processor = TP1Processor()
        processor.process_all()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()