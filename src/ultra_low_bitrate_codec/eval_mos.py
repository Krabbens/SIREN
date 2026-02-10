import argparse
import glob
import os
import pandas as pd
from speechmos import dnsmos
import librosa
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deg_dir", required=True, help="Directory with degraded/generated audio")
    args = parser.parse_args()
    
    files = glob.glob(os.path.join(args.deg_dir, "*.wav"))
    if not files:
        print("No wav files found in deg_dir")
        return

    print(f"Evaluating {len(files)} files using DNSMOS...")
    
    results = []
    for f in files:
        try:
            # Load audio using librosa (handles resampling and mono mix automatically)
            wav, sr = librosa.load(f, sr=16000)
            
            # DNSMOS run
            score = dnsmos.run(wav, sr)
            
            res = {
                "file": os.path.basename(f),
                "OVRL": score['ovrl_mos'],
                "SIG": score['sig_mos'],
                "BAK": score['bak_mos']
            }
            results.append(res)
            print(f"{res['file']}: OVRL={res['OVRL']:.2f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if results:
        df = pd.DataFrame(results)
        print("\n--- Summary ---")
        print(df[["OVRL", "SIG", "BAK"]].mean())
        df.to_csv("mos_results.csv", index=False)

if __name__ == "__main__":
    main()
