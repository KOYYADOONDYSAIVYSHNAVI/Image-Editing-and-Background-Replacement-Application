import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from diffusers import StableDiffusionInpaintPipeline
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

# Load YOLO model
def load_yolo_model(model_path='yolov8x-seg.pt'):
    return YOLO(model_path)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)  
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)  

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)  

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=0),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, output_padding=0),  # 32x32 -> 64x64
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        """Applies reparameterization trick: z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_input(z).view(-1, 128, 8, 8)  # Ensure correct reshaping
        decoded = self.decoder(z)
        return decoded, mu, logvar

# Load trained VAE
def load_vae(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Perform segmentation using YOLOv8
def segment_body_yolo(image, model):
    results = model(image)
    masks = results[0].masks
    body_masks = []
    
    if masks is not None:
        for idx, mask in enumerate(masks):
            class_id = results[0].boxes.cls[idx].item()
            if class_id == 0:
                body_masks.append(mask.data.cpu().numpy())

        if body_masks:
            body_masks = [(mask - mask.min()) / (mask.max() - mask.min() + 1e-6) for mask in body_masks]
            body_masks = [(mask * 255).astype(np.uint8) for mask in body_masks]
    
    return body_masks

# Refine masks using the PyTorch autoencoder
def refine_mask_with_autoencoder(mask, model):
    mask = mask.astype("float32") / 255.0
    mask = torch.tensor(mask).unsqueeze(0)  # Add batch dim
    if mask.ndim == 4:  # Remove extra dimension if present
        mask = mask.squeeze(1)
    mask = mask.unsqueeze(1)  # Ensure shape [batch, channel, H, W]
    
    with torch.no_grad():
        refined_mask = model(mask)
    
    refined_mask = (refined_mask.squeeze().numpy() * 255).astype(np.uint8)
    return refined_mask

def load_autoencoder(model_path):
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
# Load Stable Diffusion Inpainting Pipeline
def load_stable_diffusion():
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    pipe = pipe.to("cuda") if torch.cuda.is_available() else pipe.to("cpu")
    return pipe

# Image generation function using Stable Diffusion
def generate_image_from_prompt(prompt, image, mask_image, pipe):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(mask_image, mode="L")
    generated_image = pipe(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]
    return generated_image

# Enhance Image using Custom VAE
def enhance_image_with_custom_vae(image, vae):
    image = image.resize((64, 64))  # Resize to match VAE expected input size
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    with torch.no_grad():
        enhanced_image_tensor, _, _ = vae(image_tensor)  # Ensure correct unpacking
    
    enhanced_image = (enhanced_image_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(enhanced_image)

def debug_vae(image, vae):
    image = image.resize((64, 64))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        encoded = vae.encoder(image_tensor)
        print(f"Encoded shape: {encoded.shape}")

        mu = vae.fc_mu(encoded)
        logvar = vae.fc_logvar(encoded)
        print(f"Mu shape: {mu.shape}, Logvar shape: {logvar.shape}")

        z = vae.reparameterize(mu, logvar)
        print(f"Latent vector shape: {z.shape}")

        decoded = vae.decoder_input(z).view(-1, 128, 8, 8)
        print(f"Decoded input shape: {decoded.shape}")

        enhanced_image_tensor = vae.decoder(decoded)
        print(f"Enhanced image shape: {enhanced_image_tensor.shape}")

    return enhanced_image_tensor

# Streamlit UI
st.title("Enhanced Image Editing with Autoencoder, Inpainting & Custom VAE")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    yolo_model = load_yolo_model()
    autoencoder_model = load_autoencoder("autoencoder.pth")
    pipe = load_stable_diffusion()
    vae = load_vae("vae_celeba.pth")  # Load your custom trained VAE
    
    body_masks = segment_body_yolo(image, yolo_model)
    
    st.subheader("YOLOv8 Body Segmentation")
    if body_masks:
        refined_masks = []
        for idx, mask in enumerate(body_masks):
            refined_mask = refine_mask_with_autoencoder(mask, autoencoder_model)
            refined_masks.append(refined_mask)
            st.image(refined_mask, caption=f'Refined Mask {idx+1}', use_column_width=True, channels="GRAY")
        
        selected_mask_index = st.number_input("Select Person", min_value=1, max_value=len(refined_masks))
        user_prompt = st.text_input("Enter a prompt")
        
        if user_prompt:
            selected_mask = refined_masks[selected_mask_index - 1]
            inverted_mask = 255 - selected_mask
            st.image(inverted_mask, caption=f"Selected Mask {selected_mask_index}", use_column_width=True, channels="GRAY")
            
            # Generate image
            generated_image = generate_image_from_prompt(user_prompt, image, inverted_mask, pipe)
            st.image(generated_image, caption="Generated Image", use_column_width=True)

            debug_output = debug_vae(generated_image, vae)
            st.write("VAE Debug Output:")
            st.write(debug_output.shape)

            
            # Apply Custom VAE Enhancement
            enhanced_image = enhance_image_with_custom_vae(generated_image, vae)
            st.image(enhanced_image, caption="VAE Enhanced Image", use_column_width=True)
    else:
        st.write("No body detected.")
