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
