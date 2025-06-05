import shap
import torch
from models.efficientnet_model import ECGNet
from scripts.preprocess import load_dataset

def explain():
    model = ECGNet()
    model.load_state_dict(torch.load('outputs/efficientnet_ecg_model.pth'))
    model.eval()

    data, _ = next(iter(load_dataset(split='test', return_loader=True)))
    background = data[:100]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data[:10])

    shap.image_plot(shap_values, data[:10])

if __name__ == "__main__":
    explain()
