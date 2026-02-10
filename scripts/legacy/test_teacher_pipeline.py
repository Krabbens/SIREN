
import os
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import yaml
import soundfile as sf
from transformers import AutoModel

# Models
from ultra_low_bitrate_codec.models.flow_matching import ConditionalFlowMatching
from ultra_low_bitrate_codec.models.fuser import ConditionFuserV2
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.mel_vocoder import MelVocoderBitNet

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_wav = "data/jakubie.wav"
    output_wav = "outputs/teacher_ref_flow31.wav"
    flow_suffix = "epoch31"
    
    print("Loading Teacher Reference Pipeline...")
    teacher = AutoModel.from_pretrained("ntu-spml/distilhubert").to(device)
    teacher.eval()
    
    # Load Flow/Fuser (Same as student test)
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_large.yaml"
    with open(config_path) as f: config = yaml.safe_load(f)
    
    factorizer_dir = "checkpoints/factorizer_microhubert_finetune_v4"
    flow_dir = "checkpoints/checkpoints_flow_v2"
    
    factorizer = InformationFactorizerV2(config).to(device)
    ckpt_f = torch.load(os.path.join(factorizer_dir, "factorizer.pt"), map_location=device)
    factorizer.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt_f.items()}, strict=False)
    factorizer.eval()

    fuser = ConditionFuserV2(sem_dim=8, pro_dim=8, spk_dim=256, out_dim=512, sem_upsample=4, pro_upsample=8).to(device)
    fu_ckpt = torch.load(os.path.join(flow_dir, f"fuser_{flow_suffix}.pt"), map_location=device)
    fuser.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fu_ckpt.items()})
    fuser.eval()

    config['model']['decoder']['fusion_dim'] = 80
    config['model']['decoder']['hidden_dim'] = 512
    flow_model = ConditionalFlowMatching(config).to(device)
    fl_ckpt = torch.load(os.path.join(flow_dir, f"flow_{flow_suffix}.pt"), map_location=device)
    flow_model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in fl_ckpt.items()})
    flow_model.eval()

    sem_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    pro_vq = ResidualFSQ(levels=config['model']['fsq_levels'], num_levels=config['model']['rfsq_num_levels'], input_dim=8).to(device)
    spk_pq = ProductQuantizer(input_dim=256, num_groups=8, codes_per_group=256).to(device)

    vocoder = MelVocoderBitNet().to(device)
    voc_ckpt = torch.load("checkpoints/vocoder_mel/vocoder_latest.pt", map_location=device)
    vocoder.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in voc_ckpt.items()})
    vocoder.eval()

    # Load Audio (Apply MAX NORM like in precompute script)
    wav, sr = sf.read(input_wav)
    wav = torch.tensor(wav, dtype=torch.float32)
    if sr != 16000: wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav / (wav.abs().max() + 1e-6)
    
    with torch.no_grad():
        t_out = teacher(wav.unsqueeze(0).to(device), output_hidden_states=True)
        features = t_out.last_hidden_state
        
        sem, pro, spk = factorizer(features)
        sem_z, _, _ = sem_vq(sem)
        pro_z, _, _ = pro_vq(pro)
        spk_z, _, _ = spk_pq(spk)
        
        cond = fuser(sem_z, pro_z, spk_z, features.shape[1])
        mel = flow_model.solve_ode(cond, steps=50, solver='rk4', cfg_scale=1.0)
        
        mel = mel * 3.5 - 5.0
        audio_out = vocoder(mel)
        sf.write(output_wav, audio_out.squeeze().cpu().numpy(), 16000)
    print(f"Saved {output_wav}")

if __name__ == "__main__":
    main()
