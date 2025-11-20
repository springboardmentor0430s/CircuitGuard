# scripts/evaluate_model.py
import json
import matplotlib.pyplot as plt
import os

def plot_and_summarize():
    """
    Reads the saved training history, generates plots for accuracy and loss,
    and creates a text summary of the final results.
    """
    print("--- Generating Final Plots and Summary from Saved Training History ---")

    # 1. Load the history file
    history_path = 'outputs/training_history.json'
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{history_path}' not found.")
        print("Please run the 'train_model.py' script first to generate the history file.")
        return

    # 2. Extract data from history
    train_acc = history['train_acc']
    test_acc = history['test_acc']
    train_loss = history['train_loss']
    test_loss = history['test_loss']
    epochs = range(1, len(train_acc) + 1)

    # 3. Create Plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Training and Validation Metrics', fontsize=16)

    # Plot Loss
    ax1.plot(epochs, train_loss, 'r-', marker='o', label='Train Loss')
    ax1.plot(epochs, test_loss, 'b-', marker='o', label='Test Loss')
    ax1.set_title('Model Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'r-', marker='o', label='Train Accuracy')
    ax2.plot(epochs, test_acc, 'b-', marker='o', label='Test Accuracy')
    ax2.axhline(y=0.97, color='g', linestyle='--', label='Target (97%)')
    ax2.set_title('Model Accuracy vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Save the plots to a file
    plot_path = 'outputs/training_plots.png'
    plt.savefig(plot_path)
    print(f"\nPlots saved to '{plot_path}'")

    # 4. Create the Summary Log File
    best_val_acc = max(test_acc)
    best_epoch = test_acc.index(best_val_acc) + 1
    final_train_acc = train_acc[-1]
    final_val_acc = test_acc[-1]
    min_val_loss = min(test_loss)
    target_achieved = "YES" if best_val_acc >= 0.97 else "NO"

    summary_text = (
        f"Total Epochs: {len(epochs)}\n\n"
        f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Achieved at Epoch {best_epoch})\n"
        f"Final Training Accuracy: {final_train_acc*100:.2f}%\n"
        f"Final Validation Accuracy: {final_val_acc*100:.2f}%\n"
        f"Minimum Validation Loss: {min_val_loss:.4f}\n\n"
        f"Target Achievement (>=97%): {target_achieved}"
    )

    summary_path = 'outputs/training_history.log'
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"Summary log saved to '{summary_path}'")

    print("\n--- Training Summary ---")
    print(summary_text)

    # Finally, show the plot
    print("\nDisplaying plots. Close the plot window to exit.")
    plt.show()

if __name__ == '__main__':
    plot_and_summarize()