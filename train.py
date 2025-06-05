import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.efficientnet_model import ECGNet
from scripts.preprocess import load_dataset

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ECGNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(load_dataset(split='train'), batch_size=32, shuffle=True)
    for epoch in range(10):
        loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), 'outputs/efficientnet_ecg_model.pth')

if __name__ == "__main__":
    main()
