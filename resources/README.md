# Resources

This folder stores non-patient, reusable research resources.

Do not place private patient data, model checkpoints, or generated experiment outputs here.

## Biological Categories

Place curated biological category files in:

```text
resources/biological_categories/
```

Expected CSV format:

```csv
cell_cycle,immune_response,metabolism
MKI67,CD3D,HK2
TOP2A,CD8A,LDHA
CCNB1,CD274,PKM
```

Each column is treated as one category/token definition.

## Pathway GMT Files

Place redistributable GMT files in:

```text
resources/pathways/
```

For restricted resources such as MSigDB, prefer documenting download instructions instead of committing the files.

Useful MSigDB pages:

- Hallmark gene sets: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#H
- Reactome gene sets are under curated canonical pathways: https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp#C2
