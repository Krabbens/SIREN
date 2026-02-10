
import torch
import yaml
from ultra_low_bitrate_codec.models.encoder import InformationFactorizerV2
from ultra_low_bitrate_codec.models.decoder import SpeechDecoderV2
from ultra_low_bitrate_codec.models.quantizers import ResidualFSQ, ProductQuantizer
from ultra_low_bitrate_codec.models.entropy_coding import EntropyModel
from ultra_low_bitrate_codec.models.tiny_hubert import TinyHubert
from transformers import HubertModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    config_path = "src/ultra_low_bitrate_codec/configs/ultra200bps_tiny_adapt.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 1. SIREN Codec Components
    factorizer = InformationFactorizerV2(config)
    decoder = SpeechDecoderV2(config)
    
    sem_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['semantic']['output_dim']
    )
    
    pro_vq = ResidualFSQ(
        levels=config['model']['fsq_levels'],
        num_levels=config['model']['rfsq_num_levels'],
        input_dim=config['model']['prosody']['output_dim']
    )
    
    spk_pq = ProductQuantizer(
        input_dim=config['model']['speaker']['embedding_dim'],
        num_groups=config['model']['speaker']['num_groups'],
        codes_per_group=config['model']['speaker']['codes_per_group']
    )
    
    entropy_model = EntropyModel(config)

    siren_params = (
        count_parameters(factorizer) +
        count_parameters(decoder) +
        count_parameters(sem_vq) +
        count_parameters(pro_vq) +
        count_parameters(spk_pq) +
        count_parameters(entropy_model)
    )

    print(f"SIREN Codec (Factorizer+Decoder+Etc): {siren_params/1e6:.2f} M params ({siren_params*4/1024/1024:.2f} MB float32)")

    # Breakdown Decoder
    rec_params = count_parameters(decoder.reconstructor)
    voc_params = count_parameters(decoder.vocoder)
    print(f"  - Reconstructor: {rec_params/1e6:.2f} M params")
    print(f"  - Vocoder: {voc_params/1e6:.2f} M params")

    # 2. TinyHubert
    tiny_hubert = TinyHubert(out_dim=768, hidden_dim=384, num_layers=12)
    tiny_params = count_parameters(tiny_hubert)
    print(f"TinyHubert: {tiny_params/1e6:.2f} M params ({tiny_params*4/1024/1024:.2f} MB float32)")

    # 3. HuBERT Base
    # Approximating parameters for standard HuBERT Base
    # hubert_base = HubertModel.from_pretrained("facebook/hubert-base-ls960") # Requires internet, might fail if no cache
    # Standard base is ~95M
    hubert_base_params = 94.68 * 1e6 
    print(f"HuBERT Base (Standard): ~{hubert_base_params/1e6:.2f} M params (~{hubert_base_params*4/1024/1024:.2f} MB float32)")

    print("-" * 30)
    total_tiny = siren_params + tiny_params
    print(f"TOTAL (SIREN + TinyHubert): {total_tiny/1e6:.2f} M params ({total_tiny*4/1024/1024:.2f} MB)")

    total_base = siren_params + hubert_base_params
    print(f"TOTAL (SIREN + HuBERT Base): {total_base/1e6:.2f} M params ({total_base*4/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()
