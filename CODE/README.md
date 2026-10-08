# RankFusion: A Diagnostic Probe for Blind Docking Bottlenecks

Code for the manuscript:

*A Diagnostic Probe for Blind Docking Bottlenecks: Sorting in 
Experimental Structures versus Detection in AlphaFold3*

## Overview

This repository contains the analysis pipeline for diagnosing 
blind docking bottlenecks. RankFusion is used here as a 
training-free, fully interpretable controlled probe — not as a 
novel scoring function — to locate the dominant bottleneck in 
blind docking and to trace how it migrates as input structure 
quality degrades.

## Structure

- `src/` — core utilities (AF3 parsing, geometry, ranking)
- `pipeline/` — main docking workflows (fpocket + P2Rank)
- `analysis/` — diagnostic analyses (mechanism, probe, oracle)
- `figures/` — figure generation scripts
- `scripts/` — environment check and reporting helpers

## Installation

```bash
pip install -r requirements.txt
