import pandas as pd
import matplotlib.pyplot as plt

def gen_plot():
    AllData = pd.read_csv("data/AllData.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    # Plot 1 – rot
    axes[0].scatter(AllData.iloc[:, 8], AllData.iloc[:, 13], c='red', s=60, alpha=0.6)
    axes[0].set_title("Datensatz 1")
    axes[0].set_xlabel("Non flavanoid phenols")
    axes[0].set_ylabel("Proline")

    # Plot 2 – grün
    axes[1].scatter(AllData.iloc[:, 7], AllData.iloc[:, 10], c='green', marker='^', s=60, alpha=0.6)
    axes[1].set_title("Datensatz 2")
    axes[1].set_xlabel("Flavanoids")
    axes[1].set_ylabel("Color Intensity")

    # Plot 3 – blau
    axes[2].scatter(AllData.iloc[:, 1], AllData.iloc[:, 7], c='blue', marker='x', s=80, alpha=0.6)
    axes[2].set_title("Datensatz 3")
    axes[2].set_xlabel("Alcohol")
    axes[2].set_ylabel("Flavanoids")

    # Plot 4 – orange
    axes[3].scatter(AllData.iloc[:, 1], AllData.iloc[:, 10], c='orange', marker='*', s=80, alpha=0.6)
    axes[3].set_title("Datensatz 4")
    axes[3].set_xlabel("Alcohol")
    axes[3].set_ylabel("Color Intensity")

    # Gesamtanpassung
    for ax in axes:
        ax.grid(True, linestyle='--', color='0.75')

    plt.tight_layout()
    fig.text(0.1,-0.15,"""Datensatz 3 mit den Merkmalen 'Alcohol' und 'Flavanoids' ist aus unserer Sicht am besten geeignet.\n 
    Es sind deutliche horizontale und vertikale Abgrenzungen erkennbar, anders wie bei den anderen Datensätzen.\n
    MinLeafNodeSize=3, Änderungen bis 20 haben nichts bewirkt.
                        """)
    pdf_path = "outputs/scatterplots.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches= "tight")


    print(f"Plot erfolgreich als PDF gespeichert unter: {pdf_path}")