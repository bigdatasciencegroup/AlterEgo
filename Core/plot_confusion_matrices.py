import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def plot_confusion_matrices(training_confusion, validation_confusion, filename=None):
    training_confusion = np.array(training_confusion)
    validation_confusion = np.array(validation_confusion)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.95)

    ticks = range(len(training_confusion))

    ax1.imshow(np.clip(np.log(training_confusion) + 1, 0, np.inf), cmap='Blues')
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.set_ylabel('Actual Class Index')
    ax1.set_xlabel('Predicted Class Index')
    ax1.set_title('Training Confusion Matrix')
    for i in range(len(training_confusion)):
        for j in range(len(training_confusion[i])):
            color = 'lime' if i == j else ('red' if training_confusion[i][j] else 'gray')
            text = ax1.text(j, i, training_confusion[i][j], ha='center', va='center', color=color,
                            size=min(100//len(training_confusion), 18))

    ax2.imshow(np.clip(np.log(validation_confusion) + 1, 0, np.inf), cmap='Blues')
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(ticks)
    ax2.set_yticklabels(ticks)
    ax2.set_ylabel('Actual Class Index')
    ax2.set_xlabel('Predicted Class Index')
    ax2.set_title('Validation Confusion Matrix')
    for i in range(len(validation_confusion)):
        for j in range(len(validation_confusion[i])):
            color = 'lime' if i == j else ('red' if validation_confusion[i][j] else 'gray')
            text = ax2.text(j, i, validation_confusion[i][j], ha='center', va='center', color=color,
                            size=min(100//len(validation_confusion), 18))
            
    if filename:
        plt.savefig(filename)
        
        

training_confusion = [[219, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 219, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 213, 0, 0, 6, 0, 0, 1, 0],
                      [0, 0, 0, 220, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 217, 0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 0, 220, 0, 0, 0, 0],
                      [0, 2, 0, 0, 0, 1, 217, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 220, 0, 0],
                      [0, 0, 1, 0, 1, 5, 0, 0, 213, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 220]]
validation_confusion = [[110, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 108, 2, 0, 0, 0, 0, 0, 0, 0],
                        [0, 11, 93, 2, 0, 3, 0, 0, 1, 0],
                        [1, 0, 0, 109, 0, 0, 0, 0, 0, 0],
                        [0, 0, 3, 0, 107, 0, 0, 0, 0, 0],
                        [0, 3, 0, 0, 0, 107, 0, 0, 0, 0],
                        [0, 2, 2, 3, 0, 4, 98, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 110, 0, 0],
                        [0, 0, 0, 0, 0, 2, 0, 0, 107, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 109]]

plot_confusion_matrices(training_confusion, validation_confusion, 'figure_0_9_confusion_matrices.png')


training_confusion = [[211, 2, 7],
                      [0, 216, 4],
                      [2, 1, 217]]
validation_confusion = [[103, 0, 7],
                        [0, 110, 0],
                        [2, 0, 108]]

plot_confusion_matrices(training_confusion, validation_confusion, 'figure_012_confusion_matrices.png')


plt.show()