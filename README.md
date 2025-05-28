
# Eczema Herpeticum (EH) classification model DEMO

This repository contains [Streamlit](https://streamlit.io/) demo applications for the Eczema Herpeticum (EH) classification model. The implemented cnn model was pretrained to distinguish EH from non-EH skin conditions using clinical images.

## Files

### `demo.py`
A web app for EH classification with confidence scores and medical warnings.

**Key Features:**
- Upload a skin image and get a classification (EH or Non-EH).
- Displays model confidence for each class.
- If EH is detected, the app highlights medical urgency.

### `demo_gradcam.py`
An extended version of the classifier that includes explainability via Grad-CAM visualizations.

**Key Features:**
- Classifies uploaded clinical skin images using the same model.
- Uses Grad-CAM to generate a heatmap showing regions that influenced the model's decision (How likely those regions indicate EH?)
- Highlights confidence scores with visual cues and conditional warnings.


The pretrained cnn model weights are loaded from `best_model_trial_64.pth` and normalization values to standardize images from `mean_std_values256.npz`.

## Requirements

Install the dependencies with:

```bash
pip install streamlit torch torchvision numpy pillow opencv-python
```

Ensure the following files are present in the same directory:
- `best_model_trial_64.pth` — trained model weights
- `mean_std_values256.npz` — image-standardizing parameters
- Logo images:
  - `Eczemanet_logo.png`
  - `Imperial_College_London_new_logo.png`
  - `Logo_Main_Light.png`
- And other example EH/non-EH images

## How to Run

To launch the standard classifier:
```bash
streamlit run demo.py
```

To launch the Grad-CAM explainability app:
```bash
streamlit run demo_gradcam.py
```

## Author

Developed by Jiho Shin  
For EH classification model DEMO and integration into the Eczemanet platform by Tanaka Lab, Imperial College London
