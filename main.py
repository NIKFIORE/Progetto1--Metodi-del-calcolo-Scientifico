import numpy as np
import time
import os
from libreria import IterativeSolvers

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
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    
    print("=== Test di metodi iterativi per sistemi lineari ===")
    print("Matrici da testare:", [os.path.basename(mtx) for mtx in matrix_files])
    print("Tolleranze:", tolerances)
    
    # Esegui i test
    results = IterativeSolvers.run_tests(matrix_files, tolerances)
    
    # Stampa un riepilogo dei risultati
    print("\n=== RIEPILOGO DEI RISULTATI ===")
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

if __name__ == "__main__":
    main()