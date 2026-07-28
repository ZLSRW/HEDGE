# Data sources and provenance

The repository contains processed graph inputs derived from the resources described in the manuscript Methods.

## ASD-associated variants and regulatory links

- ASD-associated variants: NHGRI-EBI GWAS Catalog, filtered at genome-wide significance (`p ≤ 5 × 10⁻⁸`).
- Cortical rQTL links: fetal cortex eQTL and Hi-C resources, including dbGaP `phs001900.v1.p1` and `phs001190.v1.p1`, adult GTEx v8 brain cortex (`phs000424.v8.p2`), and the regulatory mapping strategy described by Golovina et al.
- hQTL links: significant H3K27ac QTL–peak associations and assigned genes from the ASD histone acetylome-wide association resource described by Sun et al.; source sequencing resources are distributed through IHEC and the repositories cited by that study.
- circQTL links: circQTL–circRNA–trans-eGene relationships from Mai et al.; Synapse accessions `syn4587609`, `syn4923029`, and `syn3275221`.

## Gene labels and interaction network

- Curated ASD genes: SFARI Gene Human Gene / Gene Scoring modules.
- Gene–gene interactions: Human Protein Atlas Interaction resource, integrating interaction evidence from IntAct, BioGRID, BioPlex, OpenCell, and associated databases.

## Processed files

The three QTL node files contain harmonized SNP identifiers, centered sequence windows, and 64-dimensional representations. `gene_nodes.csv` contains the final PPI gene universe, binary labels, label provenance, protein identifiers, and 64-dimensional gene representations. The three edge tables contain the processed LD, SNP–gene, and PPI graph structures used directly by the released code.

Access conditions for third-party source data remain governed by the originating repositories and studies. This repository distributes the processed tables included in `data/`.
