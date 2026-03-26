import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

def load_multipoles_from_folder(folder_path):
    """
    folder_path: path to folder containing text files for one model
                 each file shape: (Nk, 4) → k, P0, P2, P4
    Returns: list of arrays (Nk, 3) with P0,P2,P4 only
    """
    arrays = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            data = np.loadtxt(os.path.join(folder_path, fname))
            arrays.append(data[:,1:])  # drop k, keep P0,P2,P4
    return arrays


def plot_pca_degeneracy_cloud(model_dict):
    """
    model_dict: dictionary of model_name -> list of arrays (Nk,3) for multiple samples
                OR array of shape (Nsamples, Nk, 3)
    Plots all samples in one PCA space to show degeneracy.
    """
    all_vectors = []
    all_labels = []

    for model_name, samples in model_dict.items():
        # Ensure samples are in shape (Nsamples, Nk, 3)
        samples = np.array(samples)
        if samples.ndim == 2:
            samples = samples[None, :, :]  # add sample axis if only one

        for sample in samples:
            vec = sample.flatten()       # flatten P0,P2,P4 into single vector
            all_vectors.append(vec)
            all_labels.append(model_name)

    all_vectors = np.array(all_vectors)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(all_vectors)

    plt.style.use('dark_background')
    
    # Plot
    plt.figure(figsize=(8,6))
    colors = {'LCDM':'#1f77b4', 'nDGP':'#ff7f0e', 'f(R)':'#2ca02c', 'wCDM':'#d62728'}

    for model_name in model_dict.keys():
        mask = np.array(all_labels) == model_name
        plt.scatter(X_pca[mask,0], X_pca[mask,1], label=model_name, alpha=0.7, s=40, color=colors[model_name])

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA of multipoles: all samples in one plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()



models_degenplot = {
    'LCDM': load_multipoles_from_folder('LCDM'),
    'nDGP': load_multipoles_from_folder('nDGP'),
    'f(R)': load_multipoles_from_folder('fR'),
    'wCDM': load_multipoles_from_folder('wCDM')
}

plot_pca_degeneracy_cloud(models_degenplot)