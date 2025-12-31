import re
import sys
import argparse

# ---------------------------------------------------------
# Configuration (Must match trajectory_probe.py)
# ---------------------------------------------------------

REVISE_MARKERS = [
    "i was wrong", "you're right", "correction", "i retract", "i need to correct",
    "i made a mistake", "let me revise", "update:", "i cannot verify", "i'm not sure",
    "uncertain", "cannot confirm", "may be wrong",
    # Newer markers
    "might be some confusion", "appear to be associated", "potential assumptions", "does not correspond"
]

DOUBLEDOWN_MARKERS = [
    "as i said", "as stated", "it is", "definitely", "certainly", "the answer is"
]

REGRET_BOOST = 0.35

# ---------------------------------------------------------
# Logic
# ---------------------------------------------------------

def label_revision_behavior(commit_answer: str, revise_answer: str) -> str:
    ca = commit_answer.strip().lower()
    ra = revise_answer.strip().lower()

    if any(m in ra for m in REVISE_MARKERS):
        return "R"
    
    shared = 0
    for tok in set(re.findall(r"[a-zA-Z]{4,}", ca)):
        if tok in ra:
            shared += 1
    if shared >= 6 and any(m in ra for m in DOUBLEDOWN_MARKERS):
        return "D"
    
    if "maybe" in ra or "might" in ra or "cannot" in ra or "uncertain" in ra or "not sure" in ra:
        return "W"
    
    return "W"

def parse_log_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by "=== SAMPLE"
    # The first split part is usually header/loading logs, ignore or keep distinct
    parts = re.split(r'=== SAMPLE (\d+) ===', content)
    
    # parts[0] is preamble
    # parts[1] is ID, parts[2] is body
    # parts[3] is ID, parts[4] is body...
    
    samples = []
    
    # Iterate pairs (id, body)
    for i in range(1, len(parts), 2):
        s_id = parts[i]
        body = parts[i+1]
        
        # Regex to extract fields
        # Label: W glyph_mass: 1.0 rev_potential: 0.042
        label_match = re.search(r"Label:\s+(\w+)\s+glyph_mass:\s+([\d.-]+)\s+rev_potential:\s+([\d.-]+)", body)
        
        if not label_match:
            print(f"Skipping Sample {s_id}: Could not parse metadata line")
            continue
            
        old_label = label_match.group(1)
        old_mass = float(label_match.group(2))
        rev_pot = float(label_match.group(3))
        
        # Extract Commit and Revise texts (multiline)
        # Assuming structure:
        # Commit: ...
        # Revise: ...
        # (Next sample)
        
        # We can split by "Commit:" and "Revise:"
        # Be careful if those words appear in text, but formatted usually at start of line?
        # The script prints "Commit: <text>\nRevise: <text>"
        
        # pattern: newline + Commit:
        split_c = body.split("\nCommit: ", 1)
        if len(split_c) < 2:
            print(f"Skipping Sample {s_id}: Could not find Commit block")
            continue
            
        rest = split_c[1]
        split_r = rest.split("\nRevise: ", 1)
        if len(split_r) < 2:
            print(f"Skipping Sample {s_id}: Could not find Revise block")
            continue
            
        commit_text = split_r[0]
        revise_text = split_r[1].strip()
        
        samples.append({
            "id": s_id,
            "old_label": old_label,
            "old_mass": old_mass,
            "rev_pot": rev_pot,
            "commit": commit_text,
            "revise": revise_text
        })
        
    return samples

def main():
    if len(sys.argv) < 2:
        print("Usage: python reprocess_logs.py <logfile>")
        return

    logfile = sys.argv[1]
    samples = parse_log_file(logfile)
    
    print(f"Parsed {len(samples)} samples.\n")
    
    counts = {"R": 0, "W": 0, "D": 0}
    changes = 0
    total_rev_pot = 0.0
    
    for s in samples:
        new_label = label_revision_behavior(s["commit"], s["revise"])
        
        # Recalculate mass
        # Infer base mass from old label
        base_mass = s["old_mass"]
        if s["old_label"] == "R":
            base_mass -= REGRET_BOOST
            
        # Calc new mass
        new_mass = base_mass
        if new_label == "R":
            new_mass += REGRET_BOOST
            
        s["new_label"] = new_label
        s["new_mass"] = new_mass
        
        counts[new_label] += 1
        total_rev_pot += s["rev_pot"]
        
        if new_label != s["old_label"]:
            changes += 1
            print(f"Sample {s['id']} UPDATED: {s['old_label']} -> {new_label}")
            print(f"  Old Mass: {s['old_mass']} -> New Mass: {round(new_mass, 3)}")
            # print(f"  Revise segment: {s['revise'][:100]}...")
            
    print("-" * 40)
    print(f"Processing complete.")
    print(f"Total Samples: {len(samples)}")
    print(f"Labels Changed: {changes}")
    print(f"Final Counts: {counts}")
    if samples:
        print(f"Avg Revision Potential: {total_rev_pot / len(samples):.4f}")

    # Write revised log
    outfile = logfile.replace(".txt", "_cleaned.txt")
    if outfile == logfile:
        outfile = logfile + ".cleaned"
        
    with open(outfile, 'w', encoding='utf-8') as f:
        # Write header placeholder if needed, or just start samples
        for s in samples:
            f.write(f"\n=== SAMPLE {s['id']} ===\n")
            f.write(f"Label: {s['new_label']} glyph_mass: {round(s['new_mass'], 3)} rev_potential: {s['rev_pot']}\n")
            f.write(f"Commit: {s['commit']}\n")
            f.write(f"Revise: {s['revise']}\n")
            
        f.write(f"\nCounts: {counts}\n")
        if samples:
            f.write(f"Avg revision_potential: {total_rev_pot / len(samples)}\n")
            
    print(f"\nSaved revised results to: {outfile}")

if __name__ == "__main__":
    main()
