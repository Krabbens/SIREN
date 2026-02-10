import torch
from ultra_low_bitrate_codec.models.discriminator import HiFiGANDiscriminator
from ultra_low_bitrate_codec.training.losses import discriminator_loss, generator_loss, feature_matching_loss

def text_gan():
    print("Initializing Discriminator...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    disc = HiFiGANDiscriminator().to(device)
    print("Discriminator initialized.")
    
    # Dummy data
    B, T = 4, 16000 # 1 sec audio
    y = torch.randn(B, 1, T).to(device)
    y_hat = torch.randn(B, 1, T).to(device)
    
    print("Running Forward...")
    # Detached (for D update)
    mpd_res, mrd_res = disc(y, y_hat.detach())
    print("Forward Complete.")
    
    # Loss
    loss_d_mpd, _, _ = discriminator_loss(mpd_res[0], mpd_res[1])
    loss_d_mrd, _, _ = discriminator_loss(mrd_res[0], mrd_res[1])
    loss_d = loss_d_mpd + loss_d_mrd
    print(f"D Loss: {loss_d.item()}")
    
    loss_d.backward()
    print("Backward D Complete.")
    
    # Generator Logic
    mpd_res, mrd_res = disc(y, y_hat)
    loss_g_mpd, _ = generator_loss(mpd_res[1]) 
    loss_g = loss_g_mpd
    print(f"G Loss: {loss_g.item()}")
    
    loss_g.backward()
    print("Backward G Complete.")
    
if __name__ == "__main__":
    text_gan()
