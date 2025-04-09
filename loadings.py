{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 1228,
     "status": "ok",
     "timestamp": 1738917802493,
     "user": {
      "displayName": "hua Yu",
      "userId": "07626386458469956541"
     },
     "user_tz": -120
    },
    "id": "nhfuGzfyTg-i",
    "outputId": "af829c68-ef0f-42cc-d941-66ba599aec61"
   },
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount('/content/drive/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "id": "zoHTbUQCNgoh"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load the matrix from the CSV file\n",
    "file_path = \"   \" #the path for the pedigree matrix\n",
    "matrixc = pd.read_csv(file_path, header=None).values  # Assuming no header\n",
    "\n",
    "# Compute eigenvalues and eigenvectors\n",
    "eigenvalues, eigenvectors = np.linalg.eig(matrixc)\n",
    "\n",
    "# Sort eigenvalues in descending order and reorder eigenvectors accordingly\n",
    "sorted_indices = np.argsort(eigenvalues)[::-1]  # Descending order\n",
    "eigenvalues = eigenvalues[sorted_indices]\n",
    "eigenvectors = eigenvectors[:, sorted_indices]\n",
    "\n",
    "# Normalize eigenvalues by the maximum eigenvalue\n",
    "eigenvalues_normalized = eigenvalues / np.max(eigenvalues)\n",
    "\n",
    "# Identify the cutoff point (inflection point) – This needs careful manual/visual inspection.\n",
    "# You'll have to run the plot below and visually determine where the eigenvalues drop off on your datasets.\n",
    "# Example cutoff (mice data):\n",
    "cutoff_index = 169  # Replace with the actual cutoff index you observe from the plot.\n",
    "\n",
    "# Keep only the significant eigenvalues and corresponding eigenvectors\n",
    "eigenvalues_significant = eigenvalues[0:cutoff_index]\n",
    "eigenvectors_significant = eigenvectors[:, 0:cutoff_index]\n",
    "\n",
    "# Check for negative eigenvalues among the significant ones\n",
    "negative_eigenvalues_significant = eigenvalues_significant[eigenvalues_significant < 0]\n",
    "\n",
    "\n",
    "# Compute loadings, handling negative eigenvalues appropriately\n",
    "if len(negative_eigenvalues_significant) > 0:\n",
    "    print(\"Warning: Negative eigenvalues detected (among significant ones)! Loadings cannot be fully computed.\")\n",
    "    loadings = None  # Cannot compute square root of negative numbers\n",
    "else:\n",
    "    loadings = eigenvectors_significant @ np.diag(np.sqrt(eigenvalues_significant))\n",
    "\n",
    "\n",
    "# Print computed loadings\n",
    "if loadings is not None:\n",
    "    print(\"Computed Loadings:\")\n",
    "    print(loadings)\n",
    "\n",
    "    # Save loadings to a CSV file\n",
    "    output_path = \"/content/drive/My Drive/GRN/loadings.csv\"\n",
    "    pd.DataFrame(loadings).to_csv(output_path, header=False, index=False)\n",
    "    print(f\"Loadings saved to: {output_path}\")\n",
    "\n",
    "\n",
    "# Plot sorted *normalized* eigenvalues to help determine cutoff visually\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(range(1, len(eigenvalues_normalized) + 1), eigenvalues_normalized, marker='o', linestyle='-')\n",
    "plt.xlabel(\"Eigenvalue Index\")\n",
    "plt.ylabel(\"Normalized Eigenvalue\")\n",
    "plt.title(\"Sorted Normalized Eigenvalues of Matrix C (mice data)\") #mice data is an example here\n",
    "plt.grid(True)\n",
    "plt.show()\n",
    "\n",
    "# Plot sorted eigenvalues (not normalized) to help determine cutoff visually\n",
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')\n",
    "plt.xlabel(\"Eigenvalue Index\")\n",
    "plt.ylabel(\"Eigenvalue\")\n",
    "plt.title(\"Sorted Eigenvalues of Matrix C (mice data)\") #mice data is an example here\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "authorship_tag": "ABX9TyN+oInJHtdMipFquXzBy9L6",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
