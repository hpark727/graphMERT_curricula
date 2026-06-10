"""
Converts networks_kg/validated_triples.csv into the three files
required by curriculum_generator/generate_questions.py:
  - vocab.txt
  - networks.graph  (pickled networkx.MultiGraph)
  - categories.json
"""

import csv
import json
import pickle
import networkx as nx

CSV_PATH = "/Users/haelpark/bottom-up-superintelligence/networks_kg/validated_triples.csv"
OUT_DIR  = "/Users/haelpark/bottom-up-superintelligence/curriculum_generator/data_kg/networks_kg"

# Only keep triples validated by both models
FILTER = lambda row: row["verdict_qwen"].strip() == "yes" and row["verdict_gpt"].strip() == "yes"

# Relations that express pure hierarchy/containment — filtered out during
# random walks (same role as 'belongs_to_the_category_of' in the DDB KG)
HIERARCHY_RELATIONS = {"is_a", "part_of", "has_part", "contains"}

# ---------- load & filter triples ----------
triples = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        if FILTER(row):
            triples.append((row["head"].strip(), row["relation"].strip(), row["tail"].strip()))

print(f"Triples after filtering: {len(triples)}")

# ---------- build vocab ----------
all_concepts = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
all_relations = sorted({r for _, r, _ in triples})
concept2id = {c: i for i, c in enumerate(all_concepts)}

print(f"Unique concepts: {len(all_concepts)}")
print(f"Unique relations: {len(all_relations)}")
print(f"Relations: {all_relations}")

with open(f"{OUT_DIR}/vocab.txt", "w") as f:
    f.write("\n".join(all_concepts))

# ---------- build MultiGraph ----------
# Forward edges: rel = index in all_relations
# Backward edges: rel = index + len(all_relations)
# The PathGenerator normalises rel with `rel % len(merged_relations)`,
# so backward traversals resolve to the same relation name.
g = nx.MultiGraph()
g.add_nodes_from(range(len(all_concepts)))

for head, rel, tail in triples:
    h_id = concept2id[head]
    t_id = concept2id[tail]
    r_idx = all_relations.index(rel)
    g.add_edge(h_id, t_id, rel=r_idx)
    g.add_edge(t_id, h_id, rel=r_idx + len(all_relations))

with open(f"{OUT_DIR}/networks.graph", "wb") as f:
    pickle.dump(g, f)

print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

# ---------- categories.json ----------
# Single flat category containing all concepts.
# Add domain-specific groupings here if needed.
categories = {"all": all_concepts}
with open(f"{OUT_DIR}/categories.json", "w") as f:
    json.dump(categories, f, indent=2)

print(f"\nFiles written to {OUT_DIR}")
print("\nPaste this as merged_relations in PathGenerator:")
print(all_relations)
print("\nHierarchy relations to filter in _get_k_hop_neighbors:")
print(sorted(HIERARCHY_RELATIONS & set(all_relations)))
