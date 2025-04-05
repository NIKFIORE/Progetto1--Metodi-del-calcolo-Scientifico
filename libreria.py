import numpy as np
import time
from scipy.io import mmread
from scipy.sparse import csr_matrix
from numpy.linalg import norm

class IterativeSolvers:
    """
    Libreria di metodi iterativi per la risoluzione di sistemi lineari Ax = b dove A è una matrice simmetrica e definita positiva.
    """
    
    @staticmethod
    def check_matrix_properties(A, x0, check_spd=True):
        """
        Verifica le proprietà della matrice A richieste per i metodi iterativi.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        x0 (numpy.ndarray): Vettore iniziale
        check_spd (bool): Se True, verifica che A sia definita positiva
        
        Returns:
        bool: True se tutte le proprietà sono soddisfatte, False altrimenti
        """
        M, N = A.shape
        L = len(x0)
        
        if M != N:
            print("La matrice A non è quadrata")
            return False
        elif L != M:
            print("Le dimensioni della matrice A non corrispondono alla dimensione del vettore iniziale x0")
            return False
            
        # Verifica che A sia simmetrica
        # Per matrici sparse, il confronto è più complesso
        if not (A != A.T).nnz == 0:
            print("La matrice A non è simmetrica")
            return False
       
       #manca verificare che la matrice sia positiva     

        return True
    
    @staticmethod
    def convergence_check(A, x, b, tol):
        """
        Verifica il criterio di convergenza ||Ax - b|| / ||b|| < tol.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        x (numpy.ndarray): Soluzione corrente
        b (numpy.ndarray): Termine noto
        tol (float): Tolleranza
        
        Returns:
        bool: True se il criterio è soddisfatto, False altrimenti
        float: Errore relativo
        """
        residual = A.dot(x) - b #dot è moltiplicazione tra matrici
        norm_b = np.linalg.norm(b)
        
        if norm_b == 0:
            # Caso speciale: se ||b|| = 0, usiamo ||Ax - b|| < tol
            rel_error = np.linalg.norm(residual)
        else:
            rel_error = np.linalg.norm(residual) / norm_b
            
        return rel_error < tol, rel_error
    
    @staticmethod
    def jacobi(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo di Jacobi per la risoluzione di sistemi lineari.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        b (numpy.ndarray): Termine noto
        tol (float): Tolleranza per il criterio di arresto
        max_iter (int): Numero massimo di iterazioni
        
        Returns:
        numpy.ndarray: Soluzione approssimata
        int: Numero di iterazioni eseguite
        float: Tempo di calcolo in secondi
        float: Errore relativo finale
        """
        n = len(b)
        x0 = np.zeros(n)  # Vettore iniziale nullo
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x0):
            return None, 0, 0.0, float('inf')
            
        # Verifica elementi diagonali non nulli
        diag_elements = A.diagonal()#estrazione della daigonale 
        if np.any(diag_elements == 0):
            print("Almeno un elemento della diagonale è nullo. Il metodo fallisce.")
            return None, 0, 0.0, float('inf')
        
        # Estrazione delle matrici per il metodo di Jacobi
        D_inv = 1.0 / diag_elements
        R = A - csr_matrix((diag_elements, (np.arange(n), np.arange(n))), shape=(n, n))
        
        x = np.copy(x0)
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            x_new = D_inv * (b - R.dot(x))
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x_new, b, tol)
            if converged:
                break
                
            x = np.copy(x_new)
            iterations += 1
            
        elapsed_time = time.time() - start_time
        
        if iterations == max_iter:
            print(f"Il metodo di Jacobi non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
    
    @staticmethod
    def gauss_seidel(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo di Gauss-Seidel per la risoluzione di sistemi lineari.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        b (numpy.ndarray): Termine noto
        tol (float): Tolleranza per il criterio di arresto
        max_iter (int): Numero massimo di iterazioni
        
        Returns:
        numpy.ndarray: Soluzione approssimata
        int: Numero di iterazioni eseguite
        float: Tempo di calcolo in secondi
        float: Errore relativo finale
        """
        n = len(b)
        x0 = np.zeros(n)  # Vettore iniziale nullo
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x0):
            return None, 0, 0.0, float('inf')
        
        # Estrazione delle matrici necessarie
        L = csr_matrix(np.tril(A.toarray()))
        U = A - L
        D = A.diagonal()
        
        x = np.copy(x0)
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            x_new = np.copy(x)
            for i in range(n):
                sum1 = L[i, :i].dot(x_new[:i])
                sum2 = U[i, i+1:].dot(x[i+1:])
                x_new[i] = (b[i] - sum1 - sum2) / D[i]
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x_new, b, tol)
            if converged:
                break
                
            x = np.copy(x_new)
            iterations += 1
            
        elapsed_time = time.time() - start_time
        
        if iterations == max_iter:
            print(f"Il metodo di Gauss-Seidel non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
    
    @staticmethod
    def gradient_method(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo del Gradiente per la risoluzione di sistemi lineari.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        b (numpy.ndarray): Termine noto
        tol (float): Tolleranza per il criterio di arresto
        max_iter (int): Numero massimo di iterazioni
        
        Returns:
        numpy.ndarray: Soluzione approssimata
        int: Numero di iterazioni eseguite
        float: Tempo di calcolo in secondi
        float: Errore relativo finale
        """
        n = len(b)
        x0 = np.zeros(n)  # Vettore iniziale nullo
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x0):
            return None, 0, 0.0, float('inf')
        
        x = np.copy(x0)
        r = b - A.dot(x)
        
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            Ar = A.dot(r)
            alpha = (r @ r) / (r @ Ar)
            x_new = x + alpha * r
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x_new, b, tol)
            if converged:
                break
                
            r = b - A.dot(x_new)
            x = x_new
            iterations += 1
            
        elapsed_time = time.time() - start_time
        
        if iterations == max_iter:
            print(f"Il metodo del Gradiente non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
    
    @staticmethod
    def conjugate_gradient(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo del Gradiente Coniugato per la risoluzione di sistemi lineari.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        b (numpy.ndarray): Termine noto
        tol (float): Tolleranza per il criterio di arresto
        max_iter (int): Numero massimo di iterazioni
        
        Returns:
        numpy.ndarray: Soluzione approssimata
        int: Numero di iterazioni eseguite
        float: Tempo di calcolo in secondi
        float: Errore relativo finale
        """
        n = len(b)
        x0 = np.zeros(n)  # Vettore iniziale nullo
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x0):
            return None, 0, 0.0, float('inf')
        
        x = np.copy(x0)
        r = b - A.dot(x)  # Residuo iniziale
        p = np.copy(r)  # Direzione iniziale
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            Ap = A.dot(p)
            r_dot_r = r @ r
            alpha = r_dot_r / (p @ Ap)
            
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x, b, tol)
            if converged:
                break
                
            beta = (r_new @ r_new) / r_dot_r
            p = r_new + beta * p
            
            r = r_new
            iterations += 1
            
        elapsed_time = time.time() - start_time
        
        if iterations == max_iter:
            print(f"Il metodo del Gradiente Coniugato non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
    
    @staticmethod
    def solve_system(A, b, x_exact, tol, method='all'):
        """
        Risolve un sistema lineare con uno o tutti i metodi implementati.
        
        Parametri:
        A (scipy.sparse.csr_matrix): Matrice del sistema
        b (numpy.ndarray): Termine noto
        x_exact (numpy.ndarray): Soluzione esatta
        tol (float): Tolleranza per il criterio di arresto
        method (str): Metodo da utilizzare ('jacobi', 'gauss_seidel', 'gradient', 'conjugate_gradient', 'all')
        
        Returns:
        dict: Risultati dell'esecuzione di ogni metodo
        """
        methods = {
            'jacobi': IterativeSolvers.jacobi,
            'gauss_seidel': IterativeSolvers.gauss_seidel,
            'gradient': IterativeSolvers.gradient_method,
            'conjugate_gradient': IterativeSolvers.conjugate_gradient
        }
        
        results = {}
        
        if method == 'all':
            for name, func in methods.items():
                print(f"\nRisoluzione con metodo: {name}")
                x_approx, iterations, elapsed_time, rel_err = func(A, b, tol)
                
                if x_approx is not None:
                    rel_error_solution = np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)
                    results[name] = {
                        'solution': x_approx,
                        'iterations': iterations,
                        'time': elapsed_time,
                        'residual_error': rel_err,
                        'solution_error': rel_error_solution
                    }
                    print(f"Errore relativo: {rel_error_solution}")
                    print(f"Iterazioni: {iterations}")
                    print(f"Tempo di calcolo: {elapsed_time:.6f} sec")
        else:
            if method in methods:
                func = methods[method]
                x_approx, iterations, elapsed_time, rel_err = func(A, b, tol)
                
                if x_approx is not None:
                    rel_error_solution = np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)
                    results[method] = {
                        'solution': x_approx,
                        'iterations': iterations,
                        'time': elapsed_time,
                        'residual_error': rel_err,
                        'solution_error': rel_error_solution
                    }
                    print(f"Errore relativo: {rel_error_solution}")
                    print(f"Iterazioni: {iterations}")
                    print(f"Tempo di calcolo: {elapsed_time:.6f} sec")
            else:
                print(f"Metodo '{method}' non riconosciuto.")
                
        return results
    
    @staticmethod
    def load_matrix_mtx(filename):
        """
        Carica una matrice da un file .mtx in formato sparse.
        
        Parametri:
        filename (str): Percorso del file .mtx
        
        Returns:
        scipy.sparse.csr_matrix: Matrice caricata dal file
        """
        try:
            # Legge la matrice dal file e la converte in formato csr_matrix
            matrix = mmread(filename)
            if isinstance(matrix, np.ndarray):
                return csr_matrix(matrix)
            else:
                return matrix.tocsr()  # Converti in formato CSR se non lo è già
        except Exception as e:
            print(f"Errore nel caricamento della matrice dal file {filename}: {e}")
            return None
    
    @staticmethod
    def run_tests(matrix_files, tolerances):
        """
        Esegue test su più file di matrici con diverse tolleranze.
        
        Parametri:
        matrix_files (list): Lista di percorsi di file .mtx
        tolerances (list): Lista di tolleranze da testare
        
        Returns:
        dict: Risultati per ogni matrice e tolleranza
        """
        results = {}
        
        for mtx_file in matrix_files:
            print(f"\nCaricamento matrice: {mtx_file}")
            A = IterativeSolvers.load_matrix_mtx(mtx_file)
            
            if A is None:
                continue
                
            n = A.shape[0]
            # Step 1: Creare la soluzione esatta (vettore di tutti 1)
            x_exact = np.ones(n)
            # Step 2: Calcolare il vettore b
            b = A.dot(x_exact)
            
            matrix_results = {}
            
            for tol in tolerances:
                print(f"\nTest con tolleranza: {tol}")
                matrix_results[tol] = IterativeSolvers.solve_system(A, b, x_exact, tol)
                
            results[mtx_file] = matrix_results
            
        return results