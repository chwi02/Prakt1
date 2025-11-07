import pandas as pd
import matplotlib.pyplot as plt

def gen_plot():
    AllData = pd.read_csv("data/AllData.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Aufgabe 1", fontsize=16)
    axes = axes.ravel()


    # Plot 1 – rot
    axes[0].scatter(AllData.iloc[:, 8], AllData.iloc[:, 13], c=AllData.iloc[:, 0], s=60, alpha=0.6)
    axes[0].set_title("Datensatz 1")
    axes[0].set_xlabel("Non flavanoid phenols")
    axes[0].set_ylabel("Proline")

    # Plot 2 – grün
    axes[1].scatter(AllData.iloc[:, 7], AllData.iloc[:, 10], c=AllData.iloc[:, 0], s=60, alpha=0.6)
    axes[1].set_title("Datensatz 2")
    axes[1].set_xlabel("Flavanoids")
    axes[1].set_ylabel("Color Intensity")

    # Plot 3 – blau
    axes[2].scatter(AllData.iloc[:, 1], AllData.iloc[:, 7], c=AllData.iloc[:, 0], s=80, alpha=0.6)
    axes[2].set_title("Datensatz 3")
    axes[2].set_xlabel("Alcohol")
    axes[2].set_ylabel("Flavanoids")

    # Plot 4 – orange
    axes[3].scatter(AllData.iloc[:, 1], AllData.iloc[:, 10], c=AllData.iloc[:, 0], s=80, alpha=0.6)
    axes[3].set_title("Datensatz 4")
    axes[3].set_xlabel("Alcohol")
    axes[3].set_ylabel("Color Intensity")

    # Gesamtanpassung
    for ax in axes:
        ax.grid(True, linestyle='--', color='0.75')

    plt.tight_layout()
    fig.text(0.1,-0.15,"""Da die verschiedenen Weinsorten im Datensatz 3 am ehesten getrennt vorliegen, halten wir diese Merkmale für besonders geignet.\n
             Die anderen Merkmale lassen sich mit einem CART-Algorithmus schwer klassifizieren.\n
             Anpassungen von MinLeafNodeSize zwischen 3 und 20 änderten das Ergebnis nicht wesentlich.
                        """)
    pdf_path = "outputs/aufgabe1.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches= "tight")


    print(f"Plot erfolgreich als PDF gespeichert unter: {pdf_path}")