# 🔬 DiffLinker-EGNN: Molecular Linker Generation using Diffusion Models

A deep learning project exploring **diffusion-based molecular generation** using **Equivariant Graph Neural Networks (EGNN)** for designing molecular linkers in 3D space.

---

## 🔥 Highlights

- Implemented **Diffusion Models** for molecule generation  
- Used **EGNN (Equivariant GNN)** to preserve 3D geometric structure  
- Worked with real-world **ZINC dataset**  
- Generated valid molecular linkers from disconnected fragments  
- Applied concepts from **AI + Chemistry + Graph ML**

---

## 🧠 Problem Statement

In drug discovery, molecules are often built by connecting smaller fragments.  

> **Goal:** Generate a valid molecular linker between given molecular fragments in 3D space.

---

## ⚙️ Methodology

1. **Fragment Extraction**
   - Split molecules into fragments (anchors)

2. **Graph Representation**
   - Atoms → nodes  
   - Bonds → edges  
   - Coordinates → 3D features  

3. **Diffusion Process**
   - Forward: Add noise  
   - Reverse: Denoise to generate structure  

4. **EGNN Model**
   - Preserves 3D invariance  
   - Performs message passing  

5. **Molecule Reconstruction**
   - Generates final molecule with linker  

---

## 🖼️ Architecture

![Pipeline](resources/overview.png)

---

## 🏗️ Project Structure

```bash
DiffLinker_Project/
├── DiffLinker/
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── logs/
│   ├── resources/
│   ├── src/
│   ├── generate.py
│   ├── compute_metrics.py
│   └── environment.yml
└── README.md
```

📊 Dataset
- Dataset: ZINC molecular dataset  
- Not included due to GitHub size limitations  

Download:
https://zenodo.org/record/7121271

📁 Place inside:

DiffLinker/data/zinc/
🧪 Installation
git clone https://github.com/LAKSHYAG16/DiffLinker-EGNN.git
cd DiffLinker-EGNN

conda env create -f environment.yml
conda activate difflinker
🚀 Usage
python generate.py \
  --fragments <path_to_fragments> \
  --model <model_checkpoint> \
  --linker_size <size_model>
📈 Results
Generated valid molecular linkers
Maintained 3D structural consistency
Demonstrated diffusion-based molecule generation
⚠️ Limitations
High computational cost
Large dataset dependency
Training is resource-intensive
🙌 Acknowledgement

Inspired by:

DiffLinker: Equivariant 3D-conditional Diffusion Model
https://www.nature.com/articles/s42256-024-00815-9

👨‍💻 Author

**Lakshya Garg**  
GitHub: https://github.com/LAKSHYAG16


---
