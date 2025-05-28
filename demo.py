import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import os

# -------------------------------
# Load Trained Model Definition
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
        x = torch.flatten(x, 1)
        for layer in self.fc_layers:
            x = layer(x)
        return x

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

model = Net(input_size=input_size, num_conv_layers=num_conv_layers, num_fc_layers=num_fc_layers)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# -------------------------------
# Page Setup and Styling
# -------------------------------
st.set_page_config(page_title="Eczema Herpeticum Classifier", layout="wide")

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-top: 10px;
            color: #2C3E50;
        }
        .subtitle {
            text-align: center;
            font-size: 16px;
            color: #7F8C8D;
            margin-bottom: 30px;
        }
        .result-card {
            margin: 30px auto;
            padding: 25px 40px;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
            text-align: center;
            width: 60%;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            color: #2980B9;
        }
        .confidence {
            font-size: 18px;
            color: #2C3E50;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Top Logos
# -------------------------------
col1, col_spacer, col2, col3 = st.columns([2, 5, 1, 1])

with col1:
    if os.path.exists("Eczemanet_logo.png"):
        st.image("Eczemanet_logo.png", width=180)
    else:
        st.warning("Eczemanet_logo.png not found.")

with col2:
    if os.path.exists("Imperial_College_London_new_logo.png"):
        st.image("Imperial_College_London_new_logo.png", width=120)
    else:
        st.warning("Imperial_College_London_new_logo.png not found.")

with col3:
    if os.path.exists("Logo_Main_Light.png"):
        st.image("Logo_Main_Light.png", width=120)
    else:
        st.warning("Logo_Main_Light.png not found.")

# -------------------------------
# Title and Subtitle
# -------------------------------
st.markdown('<div class="main-title">Eczema Herpeticum Image Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a clinical image of skin to get a model prediction with confidence levels</div>', unsafe_allow_html=True)

# -------------------------------
# Upload and Prediction
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=False)

    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = np.argmax(prob)

    label = "🧬 Eczema Herpeticum (EH)" if pred == 1 else "✅ Non-EH"

    warning_text = ""
    if pred == 1:
        warning_text = """
        <div style='background-color:#FFCDD2;padding:20px 30px;border-radius:10px;text-align:center;margin-top:20px;'>
            <span style='font-size:18px;font-weight:bold;color:#B71C1C;'>⚠️ Medical Warning</span><br><br>
            <span style='font-size:16px;color:#C62828;'>This image is likely to be Eczema Herpeticum.<br>
            Please seek immediate medical attention at the nearest hospital or clinic.</span>
        </div>
        """

    st.markdown(f"""
        <div class="result-card">
            <div class="prediction">Prediction: {label}</div>
            <div class="confidence">
                EH Confidence: {prob[1]:.2%}<br>
                Non-EH Confidence: {prob[0]:.2%}
            </div>
        </div>
        {warning_text}
    """, unsafe_allow_html=True)