import matplotlib.pyplot as plt
import re
import numpy as np

def parse_cleaned_results(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by "=== SAMPLE"
    parts = re.split(r'=== SAMPLE (\d+) ===', content)
    
    samples = []
    # Loop over ID, Body pairs
    for i in range(1, len(parts), 2):
        s_id = parts[i]
        body = parts[i+1]
        
        label_match = re.search(r"Label:\s+(\w+)\s+glyph_mass:\s+([\d.-]+)\s+rev_potential:\s+([\d.-]+)", body)
        
        # Robust extraction of the "Revise:" block
        revise_block = body.split("\nRevise: ", 1)
        revise_text = ""
        if len(revise_block) > 1:
            revise_text = revise_block[1].strip()
        
        if label_match:
            label = label_match.group(1)
            mass = float(label_match.group(2))
            pot = float(label_match.group(3))
            samples.append({
                'label': label, 
                'mass': mass, 
                'pot': pot,
                'revise_text': revise_text
            })
            
    return samples

def calculate_depth_score(text):
    # Heuristic for "depth": length + hedge count
    length_score = len(text.split()) / 5.0 # 1 point per 5 words
    
    hedges = ["may", "might", "uncertain", "possibly", "unclear", "misunderstanding", "confusion", "however", "re-evaluate", "realize"]
    hedge_count = sum(1 for h in hedges if h in text.lower())
    
    # Scale hedge factor
    return length_score + (hedge_count * 10)

def create_viz(data_file):
    samples = parse_cleaned_results(data_file)
    
    masses = [s['mass'] for s in samples]
    pots = [s['pot'] for s in samples]
    labels = [s['label'] for s in samples]
    
    # Calculate sizes
    depths = [calculate_depth_score(s['revise_text']) for s in samples]
    # Normalize sizes for plotting (min size 20, max size 500)
    if depths:
        max_d = max(depths)
        min_d = min(depths)
        sizes = [20 + ((d - min_d) / (max_d - min_d + 1e-6)) * 480 for d in depths]
    else:
        sizes = [100] * len(samples)
    
    # Color map
    colors = {'R': '#e74c3c', 'W': '#f1c40f', 'D': '#2ecc71'} 
    c_list = [colors.get(l, '#95a5a6') for l in labels]
    
    plt.figure(figsize=(12, 8)) # Slightly larger
    
    # scatter plot
    plt.scatter(masses, pots, c=c_list, s=sizes, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    # Legend for Colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=f'Label {l}',
                          markerfacecolor=c, markersize=10) for l, c in colors.items()]
    
    # Add dummy legend for Size
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label='Size = Revise Depth',
                          markerfacecolor='gray', markersize=15, alpha=0.5))
                          
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Glyph Mass vs. Revision Potential (Size = Revision Depth)")
    plt.xlabel("Glyph Mass (Commitment Strength)")
    plt.ylabel("Revision Potential (Alignment with Prompt)")
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    output_path = "glyph_analysis_plot_depth.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    create_viz("results_cleaned.txt")
