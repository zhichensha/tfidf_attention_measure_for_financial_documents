
# Attention Measure Generation Suite

## Overview
This suite of Python scripts automates the process of generating attention scores for keywords in the oil and gas industry, based on a modified version of the TF-IDF algorithm described by Flynn and Sastry in "Attention Cycles." (Flynn, Joel P. and Sastry, Karthik, Attention Cycles (February 7, 2024). Available at SSRN: https://ssrn.com/abstract=3592107 or http://dx.doi.org/10.2139/ssrn.3592107) The methodology adjusts traditional TF-IDF to incorporate both environment and renewable energy aware attention across regulatory filings and guideline documents.

## Features
- **Keyword Extraction**: Uses a modified TF-IDF algorithm to identify and rank keywords based on their significance across corporate filings and guideline documents.
- **Automated Processing**: Automatically processes textual data from PDFs and other documents, preprocesses text, and calculates attention scores.
- **Customization**: Allows for manual input of keywords and adjusts attention scoring based on specific industry or macroeconomic factors.

## Components
1. **gen_attention.py**: Main script for generating attention scores, applicable to general industries.
2. **gen_attention_oilgas.py**: Specialized script for the oil and gas industry, incorporating specific keywords and industry documents.
3. **utils.py**: Contains utility functions for text extraction, preprocessing, and TF-IDF calculation.

## Usage
Ensure Python 3.x is installed along with necessary packages. Adjust paths and input documents as necessary for specific industry analysis. Run `gen_attention.py` or `gen_attention_oilgas.py` depending on the industry focus.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn`, `nltk`, `fitz`, `bs4`
- Appropriate document sets for guideline and regulatory filing analysis

## Installation
Install required Python packages using:
```bash
pip install numpy pandas matplotlib scikit-learn nltk PyMuPDF beautifulsoup4
```
Ensure proper setup of document paths and guideline materials.

## Example
Ideal for researchers or analysts focusing on attention dynamics in specific sectors, particularly for assessing impact based on corporate disclosures and public guidelines.

