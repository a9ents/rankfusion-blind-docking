# RankFusion: A Diagnostic Probe for Blind Docking Bottlenecks

Code for the manuscript:

*A Diagnostic Probe for Blind Docking Bottlenecks: Sorting in Experimental Structures versus Detection in AlphaFold3*

## Overview

This repository contains the complete analysis pipeline and an integrated software platform for diagnosing blind docking bottlenecks. 

In this study, RankFusion is used as a training-free, fully interpretable controlled probe — not as a novel scoring function — to locate the dominant bottleneck in blind docking and to trace how it migrates as input structure quality degrades (from experimental structures to AlphaFold3 predictions).

## The AF3 Master Platform (Graphical Interface)

To streamline the complex workflows involving structure conversion, docking, cavity analysis, and OpenMM physical interventions, we provide a unified Object-Oriented (OOP) Graphical User Interface: **`AF3_Master_Platform.py`**.

### Key Features:
- **Environment Auto-Configuration**: Automatically detects and saves paths for Vina, Obabel, Meeko, and P2Rank (`master_platform_config.json`).
- **Modular Tabs**: 
  1. **Data Conversion & Repair**: Batch CIF-to-PDB extraction and DLPacker sidechain reconstruction.
  2. **Molecular Docking**: Comprehensive Vina docking with multiple strategies (fpocket + Rank Fusion, P2Rank + Rank Fusion, Manual Box, or Blind Docking).
  3. **Probe & Cavity Analysis**: Quantitative probe radius calculation to detect steric hindrance at the true ligand center.
  4. **Physical Interventions (OpenMM)**: Automated thermodynamic interventions including cryo-warming, guided ligand insertion, and 4-layer verification.

### Quick Start
1. Install dependencies (a Conda environment is highly recommended):
   ```bash
   pip install -r requirements.txt