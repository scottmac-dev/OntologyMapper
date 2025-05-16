import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import sys
from datetime import datetime


# Load model locally for semantic embedding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Clean labels
def normalize(label: str):
    return label.strip().lower().replace("_", " ").replace("-", " ")

# Input JSON files
def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

# Output results as JSON
def save_output(mapping: dict, output_path: Path):
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

# Map source labels to their semantically similar target labels
# Threshold is minimum similarity score returned from model to consider the labels a match and apply the mapping
# If not reaching this score, labels are tagged and UNKNOWN as they don't relate to the target schema and will require new labels
def map_labels(source_labels, target_labels, threshold=0.75):
    normalized_targets = [normalize(label) for label in target_labels]
    target_embeddings = model.encode(normalized_targets, convert_to_tensor=True)

    result = {}
    for label in source_labels:
        norm_label = normalize(label)
        label_embedding = model.encode(norm_label, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(label_embedding, target_embeddings)[0] # compare source target embedding to each target embedding
        top_score, top_idx = torch.max(scores, dim=0) # get top match score        
        
        # If meets threshold apply label, else UNKNOWN
        if top_score >= threshold:
            result[label] = {
                "mapping": target_labels[top_idx], 
                "score": round(top_score.item(),3),
                }
        else:
            result[label] = {
                "mapping": "UNKNOWN", 
                "closest target": target_labels[top_idx],
                "score": round(top_score.item(), 3),
                }

    return result

# Produce report of mapping results as .txt file
def produce_report(output_path, source_len, target_len, threshold= 0.75):
    results = load_json(output_path)
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    unknown_results = 0
    for key in results:
        if results[key]["mapping"] == "UNKNOWN":
            unknown_results += 1

    success_results = len(results) - unknown_results
    success_percentage = round(success_results/len(results) * 100,2)
    indent = " " * 4
    
    report_path = Path("output/report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    
    with report_path.open("w") as f:
        f.write(f"===== START REPORT =====\n")
        f.write(f"\n")

        f.write(f"TITLE: Ontology Mapping Report {current_time}\n")
        f.write(f"\n")

        f.write(f"SUMMARY:\n")
        f.write(f"{indent}- Source labels mapped: {source_len}\n")
        f.write(f"{indent}- Target labels: {target_len}\n")
        f.write(f"{indent}- Threshold for success: {threshold}\n")
        f.write(f"{indent}- Success percentage: {success_percentage}%\n")
        f.write(f"{indent}- Labels mapped: {success_results}\n")
        f.write(f"{indent}- Labels unknown: {unknown_results}\n")
        f.write(f"\n")

        f.write(f"RESULTS:\n")
        f.write(f"{indent}== SUCCESS ==\n")
        for key in results:
            if results[key]["mapping"] != "UNKNOWN":
                f.write(f"{indent}- Source: {key.upper()} -> Target: {results[key]["mapping"].upper()} || Score: {results[key]["score"]}\n")
        
        f.write(f"\n")
        f.write(f"{indent}== UNKNOWN ==\n")
        for key in results:
            if results[key]["mapping"] == "UNKNOWN":
                f.write(f"{indent}- Source: {key.upper()} -> Target: {results[key]["mapping"].upper()} || Closest target: {results[key]["closest target"].upper()} Score: {results[key]["score"]}\n")
        f.write(f"\n")
        f.write(f"===== END REPORT =====\n")

# Run CLI, show usage prompt if used incorrectly 
if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print("Example Usage: python map_labels.py <assets/source_ontology_labels.json> <assets/target_schema_ontology.json> <output/output.json>")
        print("Optional args: <threshold between 0..1>, default 0.75")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    target_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])
    threshold = float(sys.argv[4])
    report_path= Path("output/report.txt")
    
    source_labels = load_json(source_path)
    target_labels = load_json(target_path)
    
    if threshold:
        mapping = map_labels(source_labels, target_labels, threshold)
    else:
        mapping = map_labels(source_labels, target_labels)
    
    save_output(mapping, output_path)
    
    if threshold:
        produce_report(output_path, len(source_labels), len(target_labels), threshold)
    else:
        produce_report(output_path, len(source_labels), len(target_labels))
    
    
    print(f"Mapping saved to {output_path}")
    print(f"Report saved to {report_path}")

