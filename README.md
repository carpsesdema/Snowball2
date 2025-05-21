# Snowball Annotator

Snowball Annotator is a Python-based desktop application designed for image annotation, particularly for creating datasets for object detection models like YOLO. It offers a user-friendly interface with features catering to both basic annotation needs and advanced, AI-assisted workflows.

## Features

The application supports two tiers of functionality: Basic and Pro. With the provided modifications, the application will default to **Pro Tier** features.

### Core Features (Basic Tier)

*   **Image Loading:** Load entire directories of images (PNG, JPG, BMP, GIF, TIFF, WEBP).
*   **Session Management:** Save and load annotation progress (image list, annotations, class lists) in a `.json` session file.
*   **Bounding Box Annotation:** Manually draw, resize, and move bounding boxes on images.
*   **Class Management:** Define and manage a list of object classes for annotations. Annotations for removed classes are automatically cleaned up.
*   **Navigation:** Easily navigate between images (next, previous, go to specific image).
*   **Approval Workflow:** Mark images and their annotations as "approved," which is crucial for training and export.
*   **Negative Images:** Mark images as "negative" (containing no objects of interest).
*   **YOLO Export:** Export approved annotations in the YOLO darknet format, automatically creating `dataset.yaml`, image/label directory structures, and train/validation splits.
*   **Paste Last Box:** Quickly paste the last drawn (and approved) bounding box onto the current image, centered.
*   **Legacy Settings:** A simple dialog for basic settings (e.g., confidence threshold, though mainly a Pro feature).

### Advanced Features (Pro Tier - Default)

*   **AI-Powered Suggestions:**
    *   Receive bounding box suggestions from a pre-trained/fine-tuned YOLOv8 model.
    *   Toggle suggestions on/off.
    *   Adjust the confidence threshold for displaying suggestions.
    *   Double-click a suggestion to convert it into a regular, editable annotation.
*   **Automated Training:**
    *   **Automatic Triggers:** Configure the system to automatically start fine-tuning the YOLOv8 model when a certain number of new images are approved (e.g., every 20 or 100 images).
    *   **Manual Trigger:** Force a "mini-train" session at any time with current approved data.
*   **Training Dashboard:**
    *   Visualize training progress (mAP, loss metrics) from the latest `results.csv` of a training run.
    *   Configure key training parameters:
        *   Epochs and learning rates for different trigger points (20-image, 100-image).
        *   Enable/disable automatic training triggers.
    *   Configure augmentation parameters:
        *   Vertical flip probability.
        *   Horizontal flip probability.
        *   Random rotation degrees.
    *   Quickly open the folder of the last successful training run.
*   **Model Export:** Export the fine-tuned YOLOv8 model (e.g., `yolo_finetuned.pt`) for use in other applications or deployment.
*   **Background Tasks:** Predictions and training run in background threads to keep the UI responsive.
*   **Persistent Model State:** The application tracks and uses the latest successfully trained model for suggestions and further training.

## Tech Stack

*   **Python 3.x**
*   **PyQt6:** For the graphical user interface.
*   **Ultralytics YOLO:** For AI-powered suggestions and model training (specifically seems to target YOLOv8).
*   **Matplotlib:** For plotting training metrics in the dashboard.
*   **Pandas:** For reading training results (`results.csv`).
*   **Pillow (PIL):** For image manipulation during export/dataset handling.
*   **PyYAML:** For reading/writing YAML configuration files (e.g., `dataset.yaml`).

## Prerequisites

*   Python (3.8+ recommended).
*   Access to a terminal or command prompt.
*   `pip` for installing Python packages.

## Setup & Installation

1.  **Clone the Repository (if applicable) or Download Files:**
    ```bash
    # If you have it in a git repo:
    # git clone <repository_url>
    # cd <repository_directory>
    ```
    Ensure all the Python files (`main.py`, `annotator_window.py`, `state_manager.py`, `gui.py`, `training_pipeline.py`, `workers.py`, `config.py`, `dummy_components.py`) and the `.gitignore` are in the same project directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    A `requirements.txt` is included. :
    ```
    pip install -r requirements.txt 
    - this will install all the necessary packages listed in the file.
    ```
    You can install them using pip:
    ```bash
    pip install PyQt6 ultralytics matplotlib pandas Pillow PyYAML
    ```
    *Note: `ultralytics` often installs `torch`, `torchvision`, and other dependencies automatically. If you encounter issues, you might need to install PyTorch separately according to your system's CUDA compatibility if you have a GPU: [https://pytorch.org/](https://pytorch.org/)*

## Running the Application

Once the setup is complete, run the application from your terminal:

```bash
python main.py