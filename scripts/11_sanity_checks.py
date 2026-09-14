import torch
from src.simclr import nt_xent_loss

def main():
    z1 = torch.randn(16, 128)
    z2 = torch.randn(16, 128)
    loss = nt_xent_loss(z1, z2)
    assert loss.item() == loss.item()
    print("Sanity OK. SimCLR loss:", float(loss))

if __name__ == "__main__":
    main()
