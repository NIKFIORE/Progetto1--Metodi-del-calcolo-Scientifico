import numpy as np
import time
import os
import matplotlib.pyplot as plt 
from libreria import IterativeSolvers

def plot_results(results):
    """
    Crea grafici comparativi migliorati per i metodi iterativi.
    """
    # Definisci colori e stili coerenti per i vari metodi
    method_colors = {
        'jacobi': '#3498db',         # Blu
        'gauss_seidel': '#e74c3c',   # Rosso
        'gradient': '#2ecc71',       # Verde
        'conjugate_gradient': '#9b59b6'  # Viola
    }
    
    # Per ogni matrice nel dataset
    for mtx_file, mtx_results in results.items():
        mtx_name = os.path.basename(mtx_file).split('.')[0]
        
        # 1. Creiamo un grafico che confronta tutti i metodi per diverse tolleranze
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
        fig.suptitle(f"Confronto dei metodi per matrice {mtx_name}", fontsize=16, fontweight='bold')
        
        # Prepariamo i dati per tutti i grafici
        tols = list(mtx_results.keys())
        methods = list(mtx_results[tols[0]].keys())
        
        # Prepariamo le strutture dati per i confronti
        iterations_by_method = {m: [] for m in methods}
        times_by_method = {m: [] for m in methods}
        sol_errors_by_method = {m: [] for m in methods}
        res_errors_by_method = {m: [] for m in methods}
        
        for tol in tols:
            for method in methods:
                if method in mtx_results[tol]:
                    iterations_by_method[method].append(mtx_results[tol][method]['iterations'])
                    times_by_method[method].append(mtx_results[tol][method]['time'])
                    sol_errors_by_method[method].append(mtx_results[tol][method]['solution_error'])
                    res_errors_by_method[method].append(mtx_results[tol][method]['residual_error'])
        
        # Grafici dei confronti
        axs[0, 0].set_title("Numero di iterazioni per tolleranza", fontsize=14)
        axs[0, 1].set_title("Tempo di esecuzione per tolleranza", fontsize=14)
        axs[1, 0].set_title("Errore relativo sulla soluzione", fontsize=14)
        axs[1, 1].set_title("Errore relativo sul residuo", fontsize=14)
        
        for method in methods:
            axs[0, 0].plot(tols, iterations_by_method[method], 'o-', color=method_colors[method], label=method.capitalize(), linewidth=2, markersize=8)
            axs[0, 1].plot(tols, times_by_method[method], 'o-', color=method_colors[method], label=method.capitalize(), linewidth=2, markersize=8)
            axs[1, 0].plot(tols, sol_errors_by_method[method], 'o-', color=method_colors[method], label=method.capitalize(), linewidth=2, markersize=8)
            axs[1, 1].plot(tols, res_errors_by_method[method], 'o-', color=method_colors[method], label=method.capitalize(), linewidth=2, markersize=8)
        
        # Configurazione degli assi
        for i, ax in enumerate(axs.flat):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Tolleranza', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            # Etichette asse y specifiche
            if i == 0:
                ax.set_ylabel('Numero di iterazioni', fontsize=12)
            elif i == 1:
                ax.set_ylabel('Tempo (secondi)', fontsize=12)
            elif i == 2 or i == 3:
                ax.set_ylabel('Errore relativo', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{mtx_name}_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Creiamo anche un grafico per ogni tolleranza specifica
        for tol in tols:
            methods = list(mtx_results[tol].keys())
            iterations = [mtx_results[tol][m]['iterations'] for m in methods]
            times = [mtx_results[tol][m]['time'] for m in methods]
            sol_errors = [mtx_results[tol][m]['solution_error'] for m in methods]
            res_errors = [mtx_results[tol][m]['residual_error'] for m in methods]
            
            # Rendiamo i nomi dei metodi più leggibili
            method_labels = [m.replace('_', ' ').capitalize() for m in methods]
            
            # Creazione figura
            fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
            fig.suptitle(f"Prestazioni per matrice {mtx_name} (tol={tol:.0e})", 
                        fontsize=16, fontweight='bold')
            
            # Personalizzazione colori e stili
            colors = [method_colors[m] for m in methods]
            
            # Grafico a barre per le iterazioni
            bar1 = axs[0, 0].bar(method_labels, iterations, color=colors, alpha=0.8)
            axs[0, 0].set_title("Numero di Iterazioni", fontsize=14)
            axs[0, 0].set_ylabel("Iterazioni", fontsize=12)
            axs[0, 0].set_yscale('log')  # Scala logaritmica per meglio visualizzare le differenze
            
            # Aggiungiamo i valori sopra le barre
            for bar in bar1:
                height = bar.get_height()
                axs[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
            
            # Grafico a barre per i tempi
            bar2 = axs[0, 1].bar(method_labels, times, color=colors, alpha=0.8)
            axs[0, 1].set_title("Tempo di Esecuzione", fontsize=14)
            axs[0, 1].set_ylabel("Secondi", fontsize=12)
            
            # Aggiungiamo i valori sopra le barre
            for bar in bar2:
                height = bar.get_height()
                axs[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Errore sulla soluzione
            bar3 = axs[1, 0].bar(method_labels, sol_errors, color=colors, alpha=0.8)
            axs[1, 0].set_title("Errore Relativo sulla Soluzione", fontsize=14)
            axs[1, 0].set_ylabel("Errore", fontsize=12)
            axs[1, 0].set_yscale('log')  # Scala logaritmica per meglio visualizzare le differenze
            
            # Aggiungiamo i valori sopra le barre
            for bar in bar3:
                height = bar.get_height()
                axs[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=10)
            
            # Errore sul residuo
            bar4 = axs[1, 1].bar(method_labels, res_errors, color=colors, alpha=0.8)
            axs[1, 1].set_title("Errore Relativo sul Residuo", fontsize=14)
            axs[1, 1].set_ylabel("Errore", fontsize=12)
            axs[1, 1].set_yscale('log')  # Scala logaritmica per meglio visualizzare le differenze
            
            # Aggiungiamo i valori sopra le barre
            for bar in bar4:
                height = bar.get_height()
                axs[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1e}', ha='center', va='bottom', fontsize=10)
            
            # Migliorare l'aspetto di tutti i grafici
            for ax in axs.flat:
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"{mtx_name}_tol_{tol:.0e}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
    # 3. Creiamo un grafico di confronto tra le matrici per ogni metodo
    for method in methods:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
        fig.suptitle(f"Confronto matrici per metodo {method.capitalize()}", fontsize=16, fontweight='bold')
        
        matrix_names = [os.path.basename(mtx).split('.')[0] for mtx in results.keys()]
        
        for i, mtx_file in enumerate(results.keys()):
            mtx_name = matrix_names[i]
            mtx_results = results[mtx_file]
            
            iterations = []
            times = []
            sol_errors = []
            res_errors = []
            
            for tol in tols:
                if method in mtx_results[tol]:
                    iterations.append(mtx_results[tol][method]['iterations'])
                    times.append(mtx_results[tol][method]['time'])
                    sol_errors.append(mtx_results[tol][method]['solution_error'])
                    res_errors.append(mtx_results[tol][method]['residual_error'])
            
            # Plot dei risultati
            axs[0, 0].plot(tols, iterations, 'o-', label=mtx_name, linewidth=2, markersize=8)
            axs[0, 1].plot(tols, times, 'o-', label=mtx_name, linewidth=2, markersize=8)
            axs[1, 0].plot(tols, sol_errors, 'o-', label=mtx_name, linewidth=2, markersize=8)
            axs[1, 1].plot(tols, res_errors, 'o-', label=mtx_name, linewidth=2, markersize=8)
        
        axs[0, 0].set_title("Numero di iterazioni per tolleranza", fontsize=14)
        axs[0, 1].set_title("Tempo di esecuzione per tolleranza", fontsize=14)
        axs[1, 0].set_title("Errore relativo sulla soluzione", fontsize=14)
        axs[1, 1].set_title("Errore relativo sul residuo", fontsize=14)
        
        # Configurazione degli assi
        for i, ax in enumerate(axs.flat):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Tolleranza', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            
            # Etichette asse y specifiche
            if i == 0:
                ax.set_ylabel('Numero di iterazioni', fontsize=12)
            elif i == 1:
                ax.set_ylabel('Tempo (secondi)', fontsize=12)
            elif i == 2 or i == 3:
                ax.set_ylabel('Errore relativo', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{method}_matrix_comparison.png", dpi=300, bbox_inches='tight')
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
    #, 1e-6, 1e-8, 1e-10
    
    print("========= Test di metodi iterativi per sistemi lineari =========")
    print("Matrici da testare:", [os.path.basename(mtx) for mtx in matrix_files])
    print("Tolleranze:", tolerances)
    
    # Esegui i test
    results = IterativeSolvers.run_tests(matrix_files, tolerances)
    
    # Stampa un riepilogo dei risultati in formato tabella
    print("\n=========  RIEPILOGO DEI RISULTATI =========")
    
    for mtx_file, mtx_results in results.items():
        mtx_name = os.path.basename(mtx_file)
        
        for tol, methods_results in mtx_results.items():
            print(f"\n RISULTATI FINALI  - {mtx_name} (tol={tol:.0e})")
            
            # Definisci l'intestazione della tabella con allineamento corretto
            print("Metodo              | Iterazioni | Tempo (s) | Errore Finale | Errore Relativo")
            print("-" * 75)
            
            # Definisci l'ordine dei metodi nella tabella
            methods_order = ['jacobi', 'gauss_seidel', 'gradient', 'conjugate_gradient']
            method_display_names = {
                'jacobi': 'Jacobi',
                'gauss_seidel': 'Gauss-Seidel',
                'gradient': 'Gradiente',
                'conjugate_gradient': 'Gradiente Coniugato'
            }
            
            for method_key in methods_order:
                if method_key in methods_results:
                    result = methods_results[method_key]
                    method_name = method_display_names[method_key]
                    
                    # Formatta i valori per l'output con allineamento corretto
                    iterations = str(result['iterations']).rjust(10)#serve per allineare a destra una stringa, aggiungendo spazi 
                    time = f"{result['time']:.4f}".rjust(9)
                    sol_error = f"{result['solution_error']:.2e}".rjust(13)
                    res_error = f"{result['residual_error']:.2e}".rjust(15)
                    
                    # Metodo con larghezza fissa per allineamento
                    method_padded = method_name.ljust(20)
                    
                    # Stampa la riga della tabella con allineamento corretto
                    print(f"{method_padded}| {iterations} | {time} | {sol_error} | {res_error}")
    
    # Mostra i grafici
    #plot_results(results)


if __name__ == "__main__":
    main()
