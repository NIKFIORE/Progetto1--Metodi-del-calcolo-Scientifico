import numpy as np
import time
import os
import matplotlib.pyplot as plt 
from libreria import IterativeSolvers

def plot_results(results):
    """
    Crea grafici comparativi per i metodi iterativi.
    """
    for mtx_file, mtx_results in results.items():
        for tol, methods_results in mtx_results.items():
            methods = list(methods_results.keys())
            iterations = [methods_results[m]['iterations'] for m in methods]
            times = [methods_results[m]['time'] for m in methods]
            sol_errors = [methods_results[m]['solution_error'] for m in methods]
            res_errors = [methods_results[m]['residual_error'] for m in methods]
            
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Risultati per {os.path.basename(mtx_file)} (tol={tol})", fontsize=14)

            axs[0, 0].bar(methods, iterations, color='skyblue')
            axs[0, 0].set_title("Numero di Iterazioni")
            axs[0, 0].set_ylabel("Iterazioni")

            axs[0, 1].bar(methods, times, color='orange')
            axs[0, 1].set_title("Tempo di Esecuzione")
            axs[0, 1].set_ylabel("Secondi")

            axs[1, 0].bar(methods, sol_errors, color='green')
            axs[1, 0].set_title("Errore sulla Soluzione")
            axs[1, 0].set_ylabel("Errore")

            axs[1, 1].bar(methods, res_errors, color='red')
            axs[1, 1].set_title("Errore sul Residuo")
            axs[1, 1].set_ylabel("Errore")

            for ax in axs.flat:
                ax.set_xticks(range(len(methods)))
                ax.set_xticklabels([m.upper() for m in methods], rotation=45)
                ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()


def main():
    """
    Programma principale per eseguire i test sui metodi iterativi
    """
    # Percorsi dei file delle matrici nella cartella "dati"
    data_folder = "dati"
    matrix_files = [
        os.path.join(data_folder, "spa1.mtx"),
        os.path.join(data_folder, "spa2.mtx"),
        os.path.join(data_folder, "vem1.mtx"),
        os.path.join(data_folder, "vem2.mtx")
    ]
    
    # Tolleranze da testare
    tolerances = [1e-4]
    #1e-6, 1e-8, 1e-10
    
    print("========= Test di metodi iterativi per sistemi lineari =========")
    print("Matrici da testare:", [os.path.basename(mtx) for mtx in matrix_files])
    print("Tolleranze:", tolerances)
    
    # Esegui i test
    results = IterativeSolvers.run_tests(matrix_files, tolerances)
    
    # Stampa un riepilogo dei risultati
    print("\n=========  RIEPILOGO DEI RISULTATI =========")
    for mtx_file, mtx_results in results.items():
        print(f"\nRisultati per la matrice: {os.path.basename(mtx_file)}")
        
        for tol, methods_results in mtx_results.items():
            print(f"\n  Tolleranza: {tol}")
            
            for method, result in methods_results.items(): 
                print(f"    {method.upper()}:")
                print(f"      Iterazioni: {result['iterations']}")
                print(f"      Tempo: {result['time']:.6f} sec")
                print(f"      Errore soluzione: {result['solution_error']:.6e}")
                print(f"      Errore residuo: {result['residual_error']:.6e}")
    
    # Mostra i grafici
    plot_results(results)


if __name__ == "__main__":
    main()
