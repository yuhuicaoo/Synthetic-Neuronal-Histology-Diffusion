import re
import pandas as pd
from pathlib import Path

if __name__ == "__main__":
    root = Path("saves")
    results = []

    for dir in root.iterdir():
        if not dir.name.startswith('segmentation'):
            continue
        
        file = dir / "testing_results.txt"

        if not file.exists():
            continue

        lines = file.read_text().splitlines()

        match_size = re.search(r"segmentation_training_(\d+)_", dir.name)
        ds_size = int(match_size.group(1))

        if dir.name.endswith("_base"):
            dataset_type = 'base'
            gen_size = 0
        elif dir.name.endswith('_gen_2'):
            dataset_type = 'gen'
            gen_size = 2400
        elif dir.name.endswith('_gen'):
            dataset_type = 'gen'
            gen_size = 1200
        else:
            continue

        for line in lines:
            if line.strip().startswith("map "):
                columns = [x.strip() for x in line.split()]

                mean, std = columns[-2:]
                results.append({
                    "ds_size": ds_size,
                    "mean": float(mean),
                    "std": float(std),
                    "dataset_type": dataset_type,
                    "gen_size": gen_size
                })

    df = pd.DataFrame(results)
    df.to_csv("testing_results.csv", index=False)
