import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
import os

# -------------------------------
# Define CNN Model
# -------------------------------
class Net(nn.Module):
    def __init__(self, input_size, num_conv_layers, num_fc_layers):
        super(Net, self).__init__()
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        out_channels = 32
        pool_layers = [i for i in range(num_conv_layers) if i % (num_conv_layers // 6) == 0 and i != 0]

        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU(inplace=True))
            if i in pool_layers:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels += 32

        size_after_conv = input_size
        for i in range(num_conv_layers):
            size_after_conv = (size_after_conv - 3 + 2 * 1) // 1 + 1
            if i in pool_layers:
                size_after_conv //= 2

        self.fc_layers = nn.ModuleList()
        input_features = in_channels * size_after_conv * size_after_conv
        out_fc_features = 32
        for _ in range(num_fc_layers):
            self.fc_layers.append(nn.Linear(input_features, out_fc_features))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(0.5))
            input_features = out_fc_features
        self.fc_layers.append(nn.Linear(out_fc_features, 2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        self.last_conv_output = x
        x = torch.flatten(x, 1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

# -------------------------------
# Grad-CAM
# -------------------------------
def generate_gradcam(model, input_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)
        output.retain_grad()

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = [m for m in model.conv_layers if isinstance(m, nn.Conv2d)][-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(input_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    fh.remove()
    bh.remove()

    grad = gradients[0]
    act = activations[0]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze()
    cam = F.relu(cam).cpu().detach().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
    return cam

# -------------------------------
# Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 256
num_conv_layers = 6
num_fc_layers = 1

model_path = "best_model_trial_64.pth"
mean_std_file = "mean_std_values256.npz"
data = np.load(mean_std_file)
mean = data["mean"].tolist()
std = data["std"].tolist()

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

model = Net(input_size, num_conv_layers, num_fc_layers)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Eczema Herpeticum Classifier", layout="wide")

st.markdown("""
    <style>
        .main-title {text-align: center;font-size: 36px;font-weight: bold;margin-top: 10px;color: #2C3E50;}
        .subtitle {text-align: center;font-size: 16px;color: #7F8C8D;margin-bottom: 30px;}
        .info-box {padding: 20px 30px; border-radius: 10px; text-align: center; margin-top: 20px;}
        .warning-box {background-color: #FFCDD2; color: #B71C1C;}
        .success-box {background-color: #C8E6C9; color: #256029;}
    </style>
""", unsafe_allow_html=True)

# Logos
col1, col_spacer, col2, col3 = st.columns([2, 5, 1, 1])
with col1:
    if os.path.exists("Eczemanet_logo.png"):
        st.image("Eczemanet_logo.png", width=180)
with col2:
    if os.path.exists("Imperial_College_London_new_logo.png"):
        st.image("Imperial_College_London_new_logo.png", width=120)
with col3:
    if os.path.exists("Logo_Main_Light.png"):
        st.image("Logo_Main_Light.png", width=120)

st.markdown('<div class="main-title">Eczema Herpeticum Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a clinical image to receive prediction and highlighted regions</div>', unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).detach().cpu().numpy()[0]
    pred = np.argmax(probs)

    # Display result box above image
    if pred == 1:
        st.markdown(f"""
        <div class="info-box warning-box">
            ⚠️ <strong>EH Detected</strong><br>
            EH Confidence: <strong>{probs[1]:.2%}</strong><br>
            Please seek immediate medical attention at the nearest hospital or clinic.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box success-box">
            ✅ <strong>No EH Detected</strong><br>
            EH Confidence: <strong>{probs[1]:.2%}</strong><br>
            No immediate medical attention is required.
        </div>
        """, unsafe_allow_html=True)

    # Grad-CAM Visualization (only low activation shown)
    cam = generate_gradcam(model, input_tensor, pred)
    img_resized = image.resize((input_size, input_size))
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    cam_threshold = 0.3
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET).astype(np.float32) / 255
    alpha_mask = np.where(cam < cam_threshold, 0.5, 0.0).astype(np.float32)
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)

    overlay = alpha_mask * heatmap + (1 - alpha_mask) * img_np
    overlay = np.clip(overlay, 0, 1)
    overlay_img = np.uint8(255 * overlay)

    # Show images side-by-side
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(img_np, caption="🖼 Uploaded Image", width=400)
    with col_b:
        st.image(overlay_img, caption=f"🔍 Grad-CAM (CAM < {cam_threshold})", width=400)
