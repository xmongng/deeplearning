"""Plot confusion counts immediately after evaluation, independent of benchmark."""
def plot_confusion(matrix, classes, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8))
    chart = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(chart, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel='Predicted class', ylabel='True class', title=title)
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > threshold else 'black', fontsize=8)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=160)
    plt.close(fig)
