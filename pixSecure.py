import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import hashlib
import base64
import time
import logging
from scipy.stats import entropy
from flask import Flask, request, render_template, jsonify, send_file
from io import BytesIO
from PIL import Image
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


#---------------- Static Methods and Helpers ---------------
@staticmethod
def convert_to_native_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    elif isinstance(obj, np.bool_):  # Handle NumPy boolean type
        return bool(obj)  
    elif isinstance(obj, bool):  # Handle Python boolean type
        return obj  
    else:
        return obj
# --------------- Image Preprocessing Module ---------------

class ImagePreprocessor:
    """
    Handles image preprocessing operations including grayscale conversion
    and edge detection using various algorithms.
    """
    
    def __init__(self, edge_algorithm='sobel'):
        """
        Initialize the preprocessor with specified edge detection algorithm.
        
        Args:
            edge_algorithm (str): Edge detection algorithm to use ('sobel', 'canny', 'prewitt')
        """
        self.edge_algorithm = edge_algorithm
        logger.info(f"Initialized ImagePreprocessor with {edge_algorithm} algorithm")
    
    def load_image(self, image_path):
        """
        Load an image from file path.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to load image from {image_path}")
            logger.info(f"Successfully loaded image from {image_path}")
            return img
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    def convert_to_grayscale(self, image):
        """
        Convert an image to grayscale.
        
        Args:
            image (numpy.ndarray): Input color image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        if len(image.shape) == 2:
            logger.info("Image is already grayscale")
            return image
        
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.info("Converted image to grayscale")
        return gray_img
    
    def detect_edges(self, gray_image):
        """
        Apply edge detection based on the specified algorithm.
        
        Args:
            gray_image (numpy.ndarray): Grayscale input image
            
        Returns:
            numpy.ndarray: Edge detected binary image
        """
        logger.info(f"Applying {self.edge_algorithm} edge detection")
        
        if self.edge_algorithm == 'sobel':
            # Apply Sobel operator in x and y directions
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            magnitude = cv2.magnitude(sobel_x, sobel_y)
            
            # Normalize the magnitude
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert to uint8
            magnitude = np.uint8(magnitude)
            
            # Threshold to get binary edge image
            _, binary_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            
        elif self.edge_algorithm == 'canny':
            binary_edges = cv2.Canny(gray_image, 50, 150)
            
        elif self.edge_algorithm == 'prewitt':
            # Prewitt operator
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            
            # Apply kernels
            prewitt_x = cv2.filter2D(gray_image, -1, kernelx)
            prewitt_y = cv2.filter2D(gray_image, -1, kernely)
            
            # Calculate magnitude
            magnitude = cv2.magnitude(prewitt_x.astype(np.float64), prewitt_y.astype(np.float64))
            
            # Normalize and convert to uint8
            magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            magnitude = np.uint8(magnitude)
            
            # Threshold to get binary edge image
            _, binary_edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
            
        else:
            raise ValueError(f"Unsupported edge detection algorithm: {self.edge_algorithm}")
        
        logger.info(f"Edge detection completed with {self.edge_algorithm} algorithm")
        return binary_edges
    
    def convert_to_binary_matrix(self, edge_image):
        """
        Convert edge image to binary matrix (0s and 1s).
        
        Args:
            edge_image (numpy.ndarray): Edge detected image
            
        Returns:
            numpy.ndarray: Binary matrix with 0s and 1s
        """
        # Ensure we have a binary image with values 0 and 1
        binary_matrix = np.where(edge_image > 0, 1, 0).astype(np.uint8)
        logger.info(f"Converted edge image to binary matrix with shape {binary_matrix.shape}")
        return binary_matrix

    def process(self, image_path):
        """
        Complete processing pipeline: load, convert to grayscale, detect edges,
        and convert to binary matrix.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            tuple: Original image, grayscale image, edge image, and binary matrix
        """
        original_image = self.load_image(image_path)
        grayscale_image = self.convert_to_grayscale(original_image)
        edge_image = self.detect_edges(grayscale_image)
        binary_matrix = self.convert_to_binary_matrix(edge_image)
        
        logger.info("Image preprocessing completed successfully")
        return original_image, grayscale_image, edge_image, binary_matrix

# --------------- Key Generation Module ---------------

class KeyGenerator:
    """
    Generates encryption keys from binary matrices based on the feature extraction 
    method described in the research paper.
    """
    
    def __init__(self, window_size=3):
        """
        Initialize the key generator with specified window size.
        
        Args:
            window_size (int): Size of the window for feature extraction (default: 3)
        """
        self.window_size = window_size
        logger.info(f"Initialized KeyGenerator with window size {window_size}")
        
    def divide_into_windows(self, binary_matrix):
        """
        Divide binary matrix into windows of specified size.
        
        Args:
            binary_matrix (numpy.ndarray): Binary matrix of image edges
            
        Returns:
            list: List of windows (each as a numpy array)
        """
        height, width = binary_matrix.shape
        windows = []
        
        # Pad the matrix if dimensions are not divisible by window_size
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        
        if pad_height > 0 or pad_width > 0:
            padded_matrix = np.pad(binary_matrix, ((0, pad_height), (0, pad_width)), 'constant')
            logger.info(f"Padded binary matrix from {binary_matrix.shape} to {padded_matrix.shape}")
            binary_matrix = padded_matrix
            height, width = binary_matrix.shape
            
        # Extract windows
        for i in range(0, height, self.window_size):
            for j in range(0, width, self.window_size):
                window = binary_matrix[i:i+self.window_size, j:j+self.window_size]
                if window.shape == (self.window_size, self.window_size):
                    windows.append(window)
                    
        logger.info(f"Divided binary matrix into {len(windows)} windows of size {self.window_size}x{self.window_size}")
        return windows
    
    def windows_to_integers(self, windows):
        """
        Convert each window to an integer value.
        
        Args:
            windows (list): List of binary windows
            
        Returns:
            list: List of integer values
        """
        integer_values = []
        
        for window in windows:
            # Flatten the window to a 1D array
            flat_window = window.flatten()
            
            # Convert binary array to integer
            # For a 3x3 window, values range from 0 to 511 (2^9 - 1)
            integer = 0
            for bit in flat_window:
                integer = (integer << 1) | bit
                
            integer_values.append(integer)
            
        logger.info(f"Converted {len(windows)} windows to integer values")
        return integer_values
    
    def compute_averaged_key(self, integer_values):
        """
        Compute the average of adjacent integer values to generate the final key.
        
        Args:
            integer_values (list): List of integer values from windows
            
        Returns:
            int: Averaged integer value for key generation
        """
        # Ensure we have at least one value
        if not integer_values:
            logger.error("No integer values available for key generation")
            raise ValueError("No integer values available for key generation")
            
        # If only one value, return it directly
        if len(integer_values) == 1:
            logger.info("Only one integer value, returning it directly")
            return integer_values[0]
            
        # Sum adjacent values as described in the paper
        adjacent_sums = []
        for i in range(0, len(integer_values) - 2, 3):
            # Sum three adjacent values
            if i + 2 < len(integer_values):
                # Convert to int64 to avoid overflow
                adjacent_sum = int(integer_values[i]) + int(integer_values[i + 1]) + int(integer_values[i + 2])
                adjacent_sums.append(adjacent_sum)
        
        # Compute average of the sums
        if adjacent_sums:
            average_value = sum(adjacent_sums) // len(adjacent_sums)
        else:
            # If we couldn't calculate adjacent sums, use average of all integers
            average_value = sum(int(x) for x in integer_values) // len(integer_values)
            
        logger.info(f"Computed averaged key value: {average_value}")
        return average_value
    
    def integer_to_binary_key(self, average_value, key_length=256):
        """
        Convert the averaged integer value to a binary key of specified length.
        
        Args:
            average_value (int): Averaged integer value
            key_length (int): Desired key length in bits
            
        Returns:
            str: Binary key string
        """
        # Convert to binary string
        binary_str = bin(average_value)[2:]  # Remove '0b' prefix
        
        # Ensure the binary string is of desired length through hashing if needed
        if len(binary_str) < key_length:
            # Use a hashing function to extend the key
            hash_input = str(average_value).encode('utf-8')
            hash_output = hashlib.sha256(hash_input).digest()
            
            # Convert hash output to binary string
            extended_binary = ''.join(format(byte, '08b') for byte in hash_output)
            
            # Use the extended binary string
            binary_str = extended_binary[:key_length]
            
        elif len(binary_str) > key_length:
            # Truncate if too long
            binary_str = binary_str[:key_length]
            
        logger.info(f"Generated binary key of length {len(binary_str)}")
        return binary_str
    
    def convert_binary_to_key_bytes(self, binary_key):
        """
        Convert binary key string to bytes for use in encryption algorithms.
        
        Args:
            binary_key (str): Binary key string
            
        Returns:
            bytes: Key in bytes format suitable for encryption
        """
        # Ensure the binary key length is a multiple of 8
        padding_needed = (8 - len(binary_key) % 8) % 8
        padded_binary = binary_key + '0' * padding_needed
        
        # Convert binary string to bytes
        key_bytes = bytearray()
        for i in range(0, len(padded_binary), 8):
            byte = padded_binary[i:i+8]
            key_bytes.append(int(byte, 2))
            
        logger.info(f"Converted binary key to {len(key_bytes)} bytes")
        return bytes(key_bytes)
    
    def generate_key(self, binary_matrix, key_size=32):
        """
        Generate a cryptographic key from a binary matrix.
        
        Args:
            binary_matrix (numpy.ndarray): Binary matrix from edge detection
            key_size (int): Size of the key in bytes (default: 32 for 256-bit key)
            
        Returns:
            tuple: (binary_key, key_bytes) - Binary key string and key in bytes format
        """
        windows = self.divide_into_windows(binary_matrix)
        integer_values = self.windows_to_integers(windows)
        average_value = self.compute_averaged_key(integer_values)
        binary_key = self.integer_to_binary_key(average_value, key_size * 8)
        key_bytes = self.convert_binary_to_key_bytes(binary_key)
        
        logger.info(f"Successfully generated {key_size * 8}-bit key from binary matrix")
        return binary_key, key_bytes
    
    def generate_iv(self, key_bytes, iv_size=16):
        """
        Generate initialization vector (IV) for encryption modes that require it.
        
        Args:
            key_bytes (bytes): Key bytes to derive IV from
            iv_size (int): Size of IV in bytes
            
        Returns:
            bytes: Initialization vector
        """
        # Use a hash function to derive IV from the key
        hash_output = hashlib.md5(key_bytes).digest()
        iv = hash_output[:iv_size]
        
        logger.info(f"Generated {iv_size * 8}-bit IV for encryption")
        return iv

# --------------- Encryption Module ---------------

class ImageEncryptor:
    """
    Handles image encryption using the generated key.
    Supports both simple XOR encryption and advanced AES encryption.
    """
    
    def __init__(self, encryption_type='xor'):
        """
        Initialize the encryptor with specified encryption type.
        
        Args:
            encryption_type (str): Type of encryption to use ('xor' or 'aes')
        """
        self.encryption_type = encryption_type
        logger.info(f"Initialized ImageEncryptor with {encryption_type} encryption")
    
    def xor_encrypt(self, image, key_bytes):
        """
        Encrypt an image using XOR operation with the provided key.
        
        Args:
            image (numpy.ndarray): Image to encrypt
            key_bytes (bytes): Key bytes for encryption
            
        Returns:
            numpy.ndarray: Encrypted image
        """
        # Make a copy of the image to avoid modifying the original
        encrypted_image = image.copy()
        
        # Convert image to bytes
        image_bytes = image.tobytes()
        
        # Create key of same length as image by repeating the key
        extended_key = bytearray()
        while len(extended_key) < len(image_bytes):
            extended_key.extend(key_bytes)
        extended_key = bytes(extended_key[:len(image_bytes)])
        
        # Perform XOR operation
        encrypted_bytes = bytearray(a ^ b for a, b in zip(image_bytes, extended_key))
        
        # Convert back to image format with the same dtype as the original
        encrypted_image = np.frombuffer(encrypted_bytes, dtype=image.dtype).reshape(image.shape)
        
        logger.info(f"XOR encryption completed on image with shape {image.shape}")
        return encrypted_image
    
    def aes_encrypt(self, image, key_bytes, iv):
        """
        Encrypt an image using AES encryption.
        
        Args:
            image (numpy.ndarray): Image to encrypt
            key_bytes (bytes): AES key (16, 24, or 32 bytes for AES-128, AES-192, or AES-256)
            iv (bytes): Initialization vector for AES
            
        Returns:
            numpy.ndarray: Encrypted image
        """
        # Ensure key is of valid length for AES
        if len(key_bytes) not in (16, 24, 32):
            # Use SHA-256 to get a 32-byte key
            key_bytes = hashlib.sha256(key_bytes).digest()
            logger.info(f"Adjusted key to 32 bytes for AES-256 encryption")
            
        # Convert image to bytes
        image_bytes = image.tobytes()
        
        # Pad the image bytes to a multiple of 16 bytes (AES block size)
        padding_length = 16 - (len(image_bytes) % 16)
        padded_bytes = image_bytes + bytes([padding_length] * padding_length)
        
        # Create AES cipher
        cipher = Cipher(
            algorithms.AES(key_bytes),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Encrypt
        encryptor = cipher.encryptor()
        encrypted_bytes = encryptor.update(padded_bytes) + encryptor.finalize()
        
        logger.info(f"AES encryption completed with {len(key_bytes) * 8}-bit key")
        
        # For image reconstruction, we'll store the shape and type info
        # in a metadata header (this is a simplification)
        # In a real system, you'd want to store this more securely
        metadata = {
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        
        return encrypted_bytes, metadata
    
    def encrypt(self, image, key_bytes):
        """
        Encrypt an image using the specified encryption method.
        
        Args:
            image (numpy.ndarray): Image to encrypt
            key_bytes (bytes): Key for encryption
            
        Returns:
            numpy.ndarray or tuple: Encrypted image or (encrypted_bytes, metadata)
        """
        if self.encryption_type == 'xor':
            return self.xor_encrypt(image, key_bytes)
        elif self.encryption_type == 'aes':
            iv = hashlib.md5(key_bytes).digest()[:16]
            return self.aes_encrypt(image, key_bytes, iv)
        else:
            raise ValueError(f"Unsupported encryption type: {self.encryption_type}")
            
    def xor_decrypt(self, encrypted_image, key_bytes):
        """
        Decrypt an XOR-encrypted image.
        
        Args:
            encrypted_image (numpy.ndarray): Encrypted image
            key_bytes (bytes): Key used for encryption
            
        Returns:
            numpy.ndarray: Decrypted image
        """
        # XOR is symmetric, so we can use the same function
        decrypted_image = self.xor_encrypt(encrypted_image, key_bytes)
        logger.info(f"XOR decryption completed on image with shape {encrypted_image.shape}")
        return decrypted_image
    
    def aes_decrypt(self, encrypted_bytes, key_bytes, iv, metadata):
        """
        Decrypt an AES-encrypted image.
        
        Args:
            encrypted_bytes (bytes): Encrypted image bytes
            key_bytes (bytes): Key used for encryption
            iv (bytes): Initialization vector
            metadata (dict): Image metadata including shape and dtype
            
        Returns:
            numpy.ndarray: Decrypted image
        """
        # Ensure key is of valid length for AES (same as in encrypt)
        if len(key_bytes) not in (16, 24, 32):
            key_bytes = hashlib.sha256(key_bytes).digest()
            
        # Create AES cipher
        cipher = Cipher(
            algorithms.AES(key_bytes),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_bytes) + decryptor.finalize()
        
        # Remove padding
        padding_length = decrypted_padded[-1]
        decrypted_bytes = decrypted_padded[:-padding_length]
        
        # Convert back to image format
        shape = metadata['shape']
        dtype = np.dtype(metadata['dtype'])
        
        decrypted_image = np.frombuffer(decrypted_bytes, dtype=dtype).reshape(shape)
        
        logger.info(f"AES decryption completed, resulting image shape: {shape}")
        return decrypted_image
    
    def decrypt(self, encrypted_data, key_bytes, metadata=None):
        """
        Decrypt an encrypted image.
        
        Args:
            encrypted_data: Encrypted image or bytes
            key_bytes (bytes): Key used for encryption
            metadata (dict, optional): Metadata needed for AES decryption
            
        Returns:
            numpy.ndarray: Decrypted image
        """
        if self.encryption_type == 'xor':
            return self.xor_decrypt(encrypted_data, key_bytes)
        elif self.encryption_type == 'aes':
            if metadata is None:
                raise ValueError("Metadata is required for AES decryption")
            iv = hashlib.md5(key_bytes).digest()[:16]
            return self.aes_decrypt(encrypted_data, key_bytes, iv, metadata)
        else:
            raise ValueError(f"Unsupported encryption type: {self.encryption_type}")

# --------------- Security Validation Module ---------------

class SecurityValidator:
    """
    Validates the security of the encryption by performing various tests
    such as entropy analysis, histogram analysis, and correlation analysis.
    """
    
    def __init__(self):
        """Initialize the security validator."""
        logger.info("Initialized SecurityValidator")
    
    def calculate_entropy(self, image):
        """
        Calculate the Shannon entropy of an image.
        Higher entropy indicates more randomness, which is desirable for encrypted images.
        
        Args:
            image (numpy.ndarray): Image to analyze
            
        Returns:
            float: Entropy value
        """
        # Flatten the image
        flat_image = image.flatten()
        
        # Calculate histogram
        hist = np.bincount(flat_image, minlength=256)
        
        # Calculate probabilities
        probabilities = hist / np.sum(hist)
        
        # Remove zero probabilities (to avoid log(0))
        probabilities = probabilities[probabilities > 0]
        
        # Calculate entropy
        entropy_value = -np.sum(probabilities * np.log2(probabilities))
        
        logger.info(f"Calculated entropy: {entropy_value}")
        return entropy_value
    
    def generate_histogram(self, image, title="Histogram"):
        """
        Generate a histogram of pixel values in the image.
        
        Args:
            image (numpy.ndarray): Image to analyze
            title (str): Title for the histogram
            
        Returns:
            tuple: (histogram, bin_edges)
        """
        # Flatten the image
        flat_image = image.flatten()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(flat_image, bins=256, range=(0, 256))
        
        logger.info(f"Generated histogram for '{title}'")
        return hist, bin_edges
    
    def calculate_correlation(self, image):
        """
        Calculate the correlation between adjacent pixels in the image.
        
        Args:
            image (numpy.ndarray): Image to analyze
            
        Returns:
            dict: Correlation coefficients in horizontal, vertical, and diagonal directions
        """
        # Flatten the image
        flat_image = image.flatten()
        
        # Calculate horizontal correlation
        horizontal_x = flat_image[:-1]
        horizontal_y = flat_image[1:]
        horizontal_corr = np.corrcoef(horizontal_x, horizontal_y)[0, 1]
        
        # Calculate vertical correlation
        height, width = image.shape
        vertical_x = image[:-1, :].flatten()
        vertical_y = image[1:, :].flatten()
        vertical_corr = np.corrcoef(vertical_x, vertical_y)[0, 1]
        
        # Calculate diagonal correlation
        diagonal_x = image[:-1, :-1].flatten()
        diagonal_y = image[1:, 1:].flatten()
        diagonal_corr = np.corrcoef(diagonal_x, diagonal_y)[0, 1]
        
        correlation_results = {
            'horizontal': horizontal_corr,
            'vertical': vertical_corr,
            'diagonal': diagonal_corr
        }
        
        logger.info(f"Calculated correlation coefficients: {correlation_results}")
        return correlation_results
    
    def calculate_npcr(self, original_encrypted, modified_encrypted):
        """
        Calculate the Number of Pixels Change Rate (NPCR).
        Measures the percentage of pixels that change value after a small change in the input.
        
        Args:
            original_encrypted (numpy.ndarray): Encrypted image from original image
            modified_encrypted (numpy.ndarray): Encrypted image from slightly modified original
            
        Returns:
            float: NPCR value (percentage)
        """
        # Create a difference map
        diff_map = np.where(original_encrypted != modified_encrypted, 1, 0)
        
        # Calculate NPCR
        npcr = 100 * np.sum(diff_map) / diff_map.size
        
        logger.info(f"Calculated NPCR: {npcr}%")
        return npcr
    
    def calculate_uaci(self, original_encrypted, modified_encrypted):
        """
        Calculate the Unified Average Changing Intensity (UACI).
        Measures the average intensity difference between two encrypted images.
        
        Args:
            original_encrypted (numpy.ndarray): Encrypted image from original image
            modified_encrypted (numpy.ndarray): Encrypted image from slightly modified original
            
        Returns:
            float: UACI value (percentage)
        """
        # Ensure images are of the same type
        original_encrypted = original_encrypted.astype(np.float32)
        modified_encrypted = modified_encrypted.astype(np.float32)
        
        # Calculate UACI
        abs_diff = np.abs(original_encrypted - modified_encrypted)
        max_value = 255.0  # For 8-bit images
        uaci = 100 * np.sum(abs_diff / max_value) / abs_diff.size
        
        logger.info(f"Calculated UACI: {uaci}%")
        return uaci
    
    def validate_encryption(self, original_image, encrypted_image):
        """
        Perform comprehensive validation of the encryption.
        
        Args:
            original_image (numpy.ndarray): Original image
            encrypted_image (numpy.ndarray): Encrypted image
            
        Returns:
            dict: Dictionary of validation results
        """
        results = {}
        
        # Entropy analysis
        original_entropy = self.calculate_entropy(original_image)
        encrypted_entropy = self.calculate_entropy(encrypted_image)
        results['entropy'] = {
            'original': original_entropy,
            'encrypted': encrypted_entropy,
            'difference': encrypted_entropy - original_entropy
        }
        
        # Histogram analysis
        original_hist, _ = self.generate_histogram(original_image, "Original Image")
        encrypted_hist, _ = self.generate_histogram(encrypted_image, "Encrypted Image")
        
        # Chi-square test for histogram uniformity
        expected_frequency = np.sum(encrypted_hist) / 256
        chi_square = np.sum((encrypted_hist - expected_frequency)**2 / expected_frequency)
        
        results['histogram'] = {
            'chi_square': chi_square,
            'original_uniformity': np.var(original_hist),
            'encrypted_uniformity': np.var(encrypted_hist)
        }
        
        # Correlation analysis
        original_corr = self.calculate_correlation(original_image)
        encrypted_corr = self.calculate_correlation(encrypted_image)
        results['correlation'] = {
            'original': original_corr,
            'encrypted': encrypted_corr
        }
        
        # Overall security assessment
        results['assessment'] = {
            'entropy_sufficient': encrypted_entropy > 7.5,  # Good encryption typically has entropy > 7.5
            'correlation_low': all(abs(v) < 0.1 for v in encrypted_corr.values()),  # Low correlation is good
            'histogram_uniform': np.var(encrypted_hist) < np.var(original_hist) * 0.1  # More uniform histogram is good
        }
        
        # Final security score (a simple measure from 0 to 10)
        entropy_score = min(10, encrypted_entropy / 8 * 10)
        correlation_score = 10 - max(abs(v) * 10 for v in encrypted_corr.values())
        histogram_score = 10 - min(10, np.var(encrypted_hist) / np.var(original_hist) * 10)
        
        results['security_score'] = (entropy_score + correlation_score + histogram_score) / 3
        
        logger.info(f"Completed security validation with score: {results['security_score']}")
        return results

# --------------- Web Application Module ---------------

app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page of the application."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and encryption."""
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image part in the request'}), 400
            
        file = request.files['image']
        
        # If user does not select file, browser may submit an empty part without filename
        if file.filename == '':
            return jsonify({'error': 'No selected image'}), 400
            
        # Check if the file is allowed
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file extension'}), 400
            
        # Save the file to a temporary location
        temp_path = os.path.join('/tmp', file.filename)
        file.save(temp_path)
        
        # Get encryption parameters from form
        edge_algorithm = request.form.get('edge_algorithm', 'sobel')
        encryption_type = request.form.get('encryption_type', 'xor')
        
        # Initialize modules
        preprocessor = ImagePreprocessor(edge_algorithm=edge_algorithm)
        key_generator = KeyGenerator()
        encryptor = ImageEncryptor(encryption_type=encryption_type)
        
        # Process the image
        original_image, grayscale_image, edge_image, binary_matrix = preprocessor.process(temp_path)
        
        # Generate key
        binary_key, key_bytes = key_generator.generate_key(binary_matrix)
        
        # Encrypt the image
        if encryption_type == 'xor':
            encrypted_image = encryptor.encrypt(grayscale_image, key_bytes)
            
            # Convert encrypted image to PNG format for display
            encrypted_buffer = BytesIO()
            encrypted_pil = Image.fromarray(encrypted_image)
            encrypted_pil.save(encrypted_buffer, format='PNG')
            encrypted_buffer.seek(0)
            
            # Encode the encrypted image in base64 for return
            encrypted_b64 = base64.b64encode(encrypted_buffer.getvalue()).decode('utf-8')
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'status': 'success',
                'key': binary_key,
                'encrypted_image': encrypted_b64
            })
            
        elif encryption_type == 'aes':
            encrypted_bytes, metadata = encryptor.encrypt(grayscale_image, key_bytes)
            
            # Store the encryption result in a session
            # Note: In a production environment, this should be stored more securely
            # For simplicity, we're returning it directly
            
            # Encode the encrypted bytes and metadata in base64 for return
            encrypted_b64 = base64.b64encode(encrypted_bytes).decode('utf-8')
            metadata_b64 = base64.b64encode(json.dumps(metadata).encode('utf-8')).decode('utf-8')
            
            # Clean up
            os.remove(temp_path)
            
            return jsonify({
                'status': 'success',
                'key': binary_key,
                'encrypted_data': encrypted_b64,
                'metadata': metadata_b64
            })
            
    except Exception as e:
        logger.error(f"Error in upload_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/decrypt', methods=['POST'])
def decrypt_image():
    
    """Handle image decryption."""
    try:
        # Get parameters from request
        key = request.form.get('key')
        encrypted_data = request.form.get('encrypted_data')
        metadata_b64 = request.form.get('metadata', None)
        encryption_type = request.form.get('encryption_type', 'xor')
        
        # Convert key to bytes
        if key.startswith('0b'):
            key = key[2:]  # Remove '0b' prefix if present
        
        # Ensure key length is a multiple of 8
        padding_needed = (8 - len(key) % 8) % 8
        padded_key = key + '0' * padding_needed
        
        # Convert binary key to bytes
        key_bytes = bytearray()
        for i in range(0, len(padded_key), 8):
            byte = padded_key[i:i+8]
            key_bytes.append(int(byte, 2))
        key_bytes = bytes(key_bytes)
        
        # Initialize encryptor
        encryptor = ImageEncryptor(encryption_type=encryption_type)
        
        # Decrypt the image
        if encryption_type == 'xor':
            # Decode the base64 encrypted image
            encrypted_bytes = base64.b64decode(encrypted_data)
            encrypted_buffer = BytesIO(encrypted_bytes)
            encrypted_image = np.array(Image.open(encrypted_buffer))
            
            # Decrypt
            decrypted_image = encryptor.decrypt(encrypted_image, key_bytes)
            
            # Convert decrypted image to PNG for display
            decrypted_buffer = BytesIO()
            decrypted_pil = Image.fromarray(decrypted_image)
            decrypted_pil.save(decrypted_buffer, format='PNG')
            decrypted_buffer.seek(0)
            
            # Return the decrypted image
            return send_file(decrypted_buffer, mimetype='image/png')
            
        elif encryption_type == 'aes':
            # Decode the base64 encrypted data and metadata
            encrypted_bytes = base64.b64decode(encrypted_data)
            metadata_str = base64.b64decode(metadata_b64).decode('utf-8')
            metadata = json.loads(metadata_str)
            
            # Decrypt
            decrypted_image = encryptor.decrypt(encrypted_bytes, key_bytes, metadata)
            
            # Convert decrypted image to PNG for display
            decrypted_buffer = BytesIO()
            decrypted_pil = Image.fromarray(decrypted_image)
            decrypted_pil.save(decrypted_buffer, format='PNG')
            decrypted_buffer.seek(0)
            
            # Return the decrypted image
            return send_file(decrypted_buffer, mimetype='image/png')
            
    except Exception as e:
        logger.error(f"Error in decrypt_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate_encryption():
    
    """Perform security validation on the encryption."""
    try:
        # Get original and encrypted images
        original_image_b64 = request.form.get('original_image')
        encrypted_image_b64 = request.form.get('encrypted_image')
        
        # Decode base64 images
        original_bytes = base64.b64decode(original_image_b64)
        encrypted_bytes = base64.b64decode(encrypted_image_b64)
        
        original_buffer = BytesIO(original_bytes)
        encrypted_buffer = BytesIO(encrypted_bytes)
        
        original_image = np.array(Image.open(original_buffer).convert('L'))  # Convert to grayscale
        encrypted_image = np.array(Image.open(encrypted_buffer))
        
        # Initialize validator
        validator = SecurityValidator()
        
        # Perform validation
        results = validator.validate_encryption(original_image, encrypted_image)

        results = convert_to_native_types(results)
        return jsonify({"status": "success", "results": results})
        
    except Exception as e:
        logger.error(f"Error in validate_encryption: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --------------- Main Entry Point ---------------

if __name__ == '__main__':
    logger.info("Starting Image Security Application")
    app.run(debug=True, host='0.0.0.0', port=5000)