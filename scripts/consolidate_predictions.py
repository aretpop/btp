import pandas as pd
import json
import glob
import os

def main():
    # Discover all predictions in experiments/baseline_*
    pred_files = glob.glob("experiments/baseline_*/predictions/*.jsonl")
    
    if not pred_files:
        print("No prediction files found!")
        return
        
    data_dict = {}
    
    for pf in pred_files:
        model_name = os.path.basename(pf).replace('.jsonl', '')
        # moondream2, qwen2-vl, internvl2-1b, florence2-base, florence2-large, etc.
        
        with open(pf, 'r') as f:
            for line in f:
                d = json.loads(line)
                image_id = d['sample_id']
                ref = d['reference_report']
                gen = d.get('generated_report', d.get('generated', ''))
                
                if image_id not in data_dict:
                    data_dict[image_id] = {
                        "image_id": image_id,
                        "reference_report": ref
                    }
                
                data_dict[image_id][f"generated_report_{model_name}"] = gen
                
    # Convert to DataFrame
    df = pd.DataFrame(list(data_dict.values()))
    
    # Save
    out_file = "consolidated_predictions.csv"
    df.to_csv(out_file, index=False)
    print(f"Consolidated predictions into {out_file} with columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
