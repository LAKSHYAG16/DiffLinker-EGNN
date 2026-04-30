# DiffLinker-EGNN: Molecular Linker Generation using Diffusion Models

This project is a simplified implementation and exploration of **DiffLinker**, an Equivariant 3D diffusion-based model for molecular linker design.

## 🧠 Overview

Molecular linker generation is an important task in drug discovery.  
Given two or more molecular fragments, the goal is to generate a valid molecule by connecting them using a chemically feasible linker.

This project uses:
- **Diffusion Models** → for generative modeling
- **EGNN (Equivariant Graph Neural Networks)** → to preserve 3D geometric structure

---

## ⚙️ Methodology

The pipeline followed in this project:

1. Fragment decomposition from molecules (ZINC dataset)
2. Conversion into graph-based molecular representation
3. Diffusion process:
   - Forward: Add noise to molecular graph
   - Reverse: Learn to denoise and generate linker
4. EGNN used for message passing and structure preservation
5. Final molecule reconstruction and evaluation

---

## 🏗️ Project Structure
