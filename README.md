Markdown

# PTEF / MANIQA - Image Quality Assessment Demo

This is the official demo code for the research project: **PTEF (Perceptual Transformer for Image Quality Assessment)**.

This code allows users to run inference on single images to predict their perceptual quality score using our trained model.

## 📂 Project Structure

- **`models/`**: Contains the model architecture definitions (PTEF/MANIQA, Swin Transformer, etc.).
- **`utils/`**: Helper functions for image preprocessing and data handling.
- **`test_images/`**: Sample images for testing.
- **`demo.py`**: The main script to run single-image inference.
- **`ptef_checkpoint.pth`**: The pre-trained model weights (A+B+C modules).
- **`requirements.txt`**: List of Python dependencies.

## 🛠️ Requirements

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
🚀 Usage
To evaluate the quality of an image, run the demo.py script.

Ensure your model weight file (ptef_checkpoint.pth) is in the root directory.

Place your test image in the directory (or use the provided samples in test_images/).

Run the following command:

Bash

python demo.py
Configuration
By default, the script looks for test_images/I15_01_1.bmp. You can modify the test_image path inside demo.py to test different images:

Python

# In demo.py
test_image = 'your_image_path.jpg' 
📊 Expected Output
The script will output the predicted quality score for the input image. (Note: This demo is for single-image inference. Full dataset evaluation (SRCC/PLCC) requires the complete dataset and validation scripts.)

Example output:

Plaintext

Using device: cuda
Initializing PTEF model architecture...
Loading weights from ptef_checkpoint.pth...
-------------------------------------------------
Image: I15_01_1.bmp
Predicted Quality Score: 0.8665
-------------------------------------------------