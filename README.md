# Image-Editing-and-Background-Replacement-Application

## Overview
This application allows users to edit images containing single or multiple people by replacing 
the background with a specified prompt-based scene. The process involves object detection, 
mask refinement, and image generation using AI models.

## Features
- **Upload an Image**: Users can upload an image containing one or more people.
- **Mask Generation**: The application detects people using YOLOv8 and generates segmentation masks.
- **Mask Refinement**: The initial masks are refined using an autoencoder.
- **User Selection and Prompt Input**: If multiple masks are detected, users can select a specific mask and provide a detailed text prompt for background replacement.
- **AI-Generated Image**: The application generates a new photorealistic image with the selected person and the specified background using Stable Diffusion from Hugging Face.
- **Image Enhancement**: The final output is enhanced using a variational autoencoder.

## Usage
1. Run the application:  
   ```sh
   streamlit run app.py
   ```
2. Upload an image containing one or multiple people.
3. The application detects people and generates masks.
4. Select a specific mask (if multiple are detected) and enter a detailed prompt (e.g., 
   *Generate a photorealistic image of the selected person, replaced into a jungle at 
   sunset, with warm golden light*).
5. Click **Generate** to process the image and replace the background.
6. The output image is enhanced and displayed.

## Technologies Used
- **YOLOv8**: Person detection and segmentation.
- **Autoencoder**: Mask refinement.
- **Stable Diffusion (Hugging Face)**: Background replacement based on user prompts.
- **Variational Autoencoder (VAE)**: Image enhancement.
- **Streamlit**: Web interface for easy user interaction.


## Autoencoder Model Architecture
The autoencoder consists of two main parts:

### Encoder (Feature Extraction & Compression)
- Three convolutional layers progressively reduce the image size.
- ReLU activation introduces non-linearity.
- Stride=2 downsamples the input, extracting key features.

### Decoder (Reconstruction)
- Three transposed convolutional layers reconstruct the original mask size.
- ReLU activation refines the output.
- Sigmoid activation ensures values are between 0 and 1.

## Training the Autoencoder
1. Run the training script:  
   ```sh
   python autoencoder.py
   ```
2. The model will train on a synthetic dataset of grayscale masks.
3. After training, the model weights will be saved as `autoencoder.pth`.

## Role of Variational Autoencoder (VAE)
Variational Autoencoder (VAE) plays a crucial role in this Streamlit Image Editing App. The  
VAE is trained on the CelebA dataset to learn latent representations of human faces, which  
allows the app to:
- Reconstruct faces with high accuracy.
- Generate mask-based facial features.
- Enable downstream tasks like enhancements.

## VAE Model Architecture
The architecture of the Variational Autoencoder consists of three main parts:

### 1. Encoder
- Takes a 64x64 RGB image and compresses it through convolutional layers.
- Outputs two vectors:  
  - μ (mu): the mean of the latent distribution  
  - log(σ²) (logvar): the log variance  
- **Architecture:**  
  - Conv2d → ReLU  
  - Conv2d → ReLU  
  - Conv2d → ReLU  
  - Flatten → Fully Connected layers for mu and logvar

### 2. Latent Space Sampling
- **Reparameterization trick:**  
  ```
  z = mu + std * eps
  ```
- Samples a point `z` from the latent space distribution.

### 3. Decoder
- Converts latent vector `z` back to an image.
- Uses transposed convolution layers to reconstruct the image.
- Outputs a 64x64 RGB image (values scaled between 0 and 1 using Sigmoid activation).

## Training the VAE

### Dataset
- **CelebA dataset** (downloaded via `torchvision.datasets.CelebA`)
- Images resized to 64x64
- Normalized to [0, 1] using `transforms.ToTensor()`

### Hyperparameters
- **Latent Dimension**: 512
- **Batch Size**: 64
- **Epochs**: 20
- **Learning Rate**: 0.001
- **Optimizer**: Adam

### Loss Function
- **Reconstruction Loss** (Mean Squared Error between input and reconstructed images)
- **KL Divergence Loss** (ensures latent vectors approximate a normal distribution)
- **Total Loss:**  
  ```
  Total Loss = Reconstruction Loss + KL Divergence Loss
  ```

### Training Loop (Summary)
1. Load CelebA dataset and preprocess images.
2. Pass batches of images through the encoder to get `mu` and `logvar`.
3. Reparameterize to get `z`.
4. Pass `z` to the decoder to get reconstructed images.
5. Calculate total loss and backpropagate.
6. Save the trained model for inference in the Streamlit app.

## Evaluation Metrics
To assess the performance of the models, the following evaluation metrics are used:
- **Intersection over Union (IoU):** Measures the overlap between predicted and ground-truth masks.
- **Mean Squared Error (MSE):** Evaluates the reconstruction quality in autoencoder-based models.
- **Structural Similarity Index (SSIM):** Compares the quality of generated images with original inputs.
- **Frechet Inception Distance (FID):** Assesses realism by comparing generated images to real ones in feature space.

## Final Results of the Project
1. The user uploads an image that may contain a single person or multiple people.
2. The YOLOv8 model segments and detects individuals in the image. If no person is detected, the system notifies the user with a **"No person detected"** message.
3. Once people are detected, the user selects a specific mask corresponding to the person they want to work with.
4. The user enters a prompt describing the desired output, and Stable Diffusion generates the image.
5. **Effectiveness of VAE Enhancement:**
   - If the prompt specifies a **simple, plain background** (e.g., *"high-quality, front-facing portrait of the chosen individual with a completely uniform, pure white background, providing a clean and distraction-free backdrop"*), the VAE enhancement works effectively. This is because the VAE was trained on the CelebA dataset, which consists of clean, front-facing, natural-looking human faces with minimal background complexity.
   - If the prompt requires a **complex or designed background** (e.g., *"jungle setting"*), the VAE enhancement does not perform as intended. The VAE struggles in these cases because it was not trained on images with diverse or highly detailed backgrounds, limiting its ability to generalize beyond the clean, portrait-style data it was trained on.

