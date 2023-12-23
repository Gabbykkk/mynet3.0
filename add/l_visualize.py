import matplotlib.pyplot as plt

def plot_heatmaps(feature1, feature2, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    im1 = axs[0].imshow(feature1, cmap='hot', interpolation='nearest')
    axs[0].set_title(title1)
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(feature2, cmap='hot', interpolation='nearest')
    axs[1].set_title(title2)
    fig.colorbar(im2, ax=axs[1])

    plt.show()
