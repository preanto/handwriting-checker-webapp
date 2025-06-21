import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# === Model Definition ===
class CVAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(784 + 10, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + 10, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

def one_hot(label, device):
    y = torch.zeros(1, 10).to(device)
    y[0][label] = 1
    return y

# === Load Model ===
device = torch.device('cpu')
model = CVAE()
model.load_state_dict(torch.load("models/cvae_mnist.pt", map_location=device))
model.eval()

# === Streamlit UI ===
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("Handwritten Digit Generator")
digit = st.selectbox("Choose a digit to generate", list(range(10)))

if st.button("Generate 5 Images"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        z = torch.randn(1, 20)
        y = one_hot(digit, device)
        with torch.no_grad():
            output = model.decode(z, y).view(28, 28).cpu()
        axs[i].imshow(output, cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
