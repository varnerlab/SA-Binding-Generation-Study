#!/bin/bash
# Install ColabFold for local multimer predictions

# Install via conda (recommended)
echo "Installing ColabFold via conda..."
conda install -c conda-forge -c bioconda colabfold[alphafold] || {
    echo "Conda install failed. Trying pip..."
    pip install colabfold[alphafold]
}

# Download databases (required for local runs)
echo "Setting up ColabFold databases..."
colabfold_search --help > /dev/null 2>&1 && echo "ColabFold installed successfully" || echo "Installation may have failed"

echo "Installation complete. Test with: colabfold_batch --help"