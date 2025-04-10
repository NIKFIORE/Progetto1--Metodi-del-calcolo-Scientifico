import numpy as np
import time
from scipy.io import mmread
from scipy.sparse import csr_matrix
from numpy.linalg import norm
from scipy import linalg

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
        # Estrae le dimensioni della matrice e del vettore
        M, N = A.shape
        L = len(x0)
        
        # Verifica che la matrice sia quadrata
        if M != N:
            print("La matrice A non è quadrata")
            return False
        # Verifica che le dimensioni della matrice e del vettore siano compatibili
        elif L != M:
            print("Le dimensioni della matrice A non corrispondono alla dimensione del vettore iniziale x0")
            return False
            
        # Verifica che A sia simmetrica
        # Per matrici sparse, verifica che la differenza tra A e la sua trasposta abbia 0 elementi non nulli
        if not (A != A.T).nnz == 0:
            print("La matrice A non è simmetrica")
            return False
       
        # Verifica elementi diagonali non nulli
        diag_elements = A.diagonal()
        if np.any(diag_elements == 0):
            print("Almeno un elemento della diagonale è nullo. Il metodo fallisce.")
            return False
            
        # Verifica che la matrice sia definita positiva usando la decomposizione di Cholesky
        # Solo se check_spd=True
        if check_spd:
            try:
                # Convertiamo la matrice in formato denso per la decomposizione di Cholesky
                A_dense = A.toarray()
                linalg.cholesky(A_dense)  # Se non è definita positiva, questa riga genererà un'eccezione
            except linalg.LinAlgError:
                print("La matrice A non è definita positiva")
                return False

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
        # Calcola il residuo r = Ax - b
        residual = A.dot(x) - b  # dot è moltiplicazione tra matrici
        norm_b = np.linalg.norm(b)  # Calcola la norma euclidea di b
        
        if norm_b == 0:
            # Caso speciale: se ||b|| = 0, usiamo ||Ax - b|| < tol
            rel_error = np.linalg.norm(residual)
        else:
            # Calcola l'errore relativo ||Ax - b|| / ||b||
            rel_error = np.linalg.norm(residual) / norm_b
            
        # Restituisce True se l'errore è minore della tolleranza
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
        x0 = np.zeros(n)  # Inizializza il vettore soluzione a zero
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x0):
            return None, 0, 0.0, float('inf')
            
        # Verifica elementi diagonali non nulli
        diag_elements = A.diagonal()
        if np.any(diag_elements == 0):
            print("Almeno un elemento della diagonale è nullo. Il metodo fallisce.")
            return None, 0, 0.0, float('inf')
        
        # Estrazione delle matrici per il metodo di Jacobi
        # D_inv è l'inversa della diagonale di A
        D_inv = 1.0 / diag_elements
        # R è la matrice A senza la diagonale (R = A - D)
        R = A - csr_matrix((diag_elements, (np.arange(n), np.arange(n))), shape=(n, n))
        
        # Inizializza il vettore soluzione
        x = np.copy(x0)
        iterations = 0
        
        # Misura il tempo di esecuzione
        start_time = time.time()
        
        # Ciclo principale del metodo di Jacobi
        while iterations < max_iter:
            # Formula di Jacobi: x^(k+1) = D^(-1) * (b - R * x^(k))
            x_new = D_inv * (b - R.dot(x))
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x_new, b, tol)
            if converged:
                break
                
            # Aggiorna il vettore soluzione per la prossima iterazione
            x = np.copy(x_new)
            iterations += 1
            
        elapsed_time = time.time() - start_time
        
        # Messaggio se non si è raggiunta la convergenza
        if iterations == max_iter:
            print(f"Il metodo di Jacobi non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
    
    @staticmethod
    def gauss_seidel_DELPROF(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo di Gauss-Seidel per la risoluzione di sistemi lineari.
        Versione precedente meno efficiente.
        
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
        
        # Estrazione delle matrici necessarie per Gauss-Seidel
        # L = matrice triangolare inferiore (inclusa la diagonale)
        L = csr_matrix(np.tril(A.toarray()))
        # U = matrice triangolare superiore (esclusa la diagonale)
        U = A - L
        # D = diagonale di A
        D = A.diagonal()
        
        x = np.copy(x0)
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            x_new = np.copy(x)
            
            # Ciclo per ogni riga della matrice
            for i in range(n):
                # Calcola la somma per la parte inferiore (già aggiornata)
                sum1 = L[i, :i].dot(x_new[:i])
                # Calcola la somma per la parte superiore (non ancora aggiornata)
                sum2 = U[i, i+1:].dot(x[i+1:])
                # Aggiorna l'i-esimo elemento di x
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
    def gauss_seidel(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo di Gauss-Seidel ottimizzato per la risoluzione di sistemi lineari.
        Questa versione è più efficiente della precedente perché accede direttamente
        ai dati della matrice sparsa.
        
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
        x = np.zeros(n)  # Vettore iniziale nullo
        
        # Verifica proprietà della matrice
        if not IterativeSolvers.check_matrix_properties(A, x):
            return None, 0, 0.0, float('inf')
        
        # Estrai i valori diagonali una sola volta per efficienza
        diag = A.diagonal()
        
        # Verifica elementi diagonali non nulli
        if np.any(diag == 0):
            print("Almeno un elemento della diagonale è nullo. Il metodo fallisce.")
            return None, 0, 0.0, float('inf')
        
        # Pre-estrai gli indici e i valori della matrice sparse
        # per un accesso più efficiente ai dati interni della matrice CSR
        A_data = A.data          # Valori non nulli
        A_indices = A.indices    # Indici di colonna per ogni valore non nullo
        A_indptr = A.indptr      # Puntatori all'inizio di ogni riga
        
        iterations = 0
        start_time = time.time()
        
        while iterations < max_iter:
            # Per ogni riga della matrice
            for i in range(n):
                # Calcola i limiti della riga i-esima nella rappresentazione CSR
                row_start, row_end = A_indptr[i], A_indptr[i+1]
                sum_val = 0.0
                
                # Calcola la somma dei termini non diagonali per la riga i
                for j_idx in range(row_start, row_end):
                    j = A_indices[j_idx]    # Indice di colonna
                    if i != j:  # Salta l'elemento diagonale
                        sum_val += A_data[j_idx] * x[j]
                
                # Aggiorna x[i] secondo la formula di Gauss-Seidel
                x[i] = (b[i] - sum_val) / diag[i]
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x, b, tol)
            if converged:
                break
                
            iterations += 1
        
        elapsed_time = time.time() - start_time
        
        if iterations == max_iter:
            print(f"Il metodo di Gauss-Seidel non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return x, iterations, elapsed_time, rel_error
     
    @staticmethod
    def gradient_method(A, b, tol=1e-6, max_iter=20000):
        """
        Metodo del Gradiente per la risoluzione di sistemi lineari.
        Cerca la soluzione nella direzione del residuo (gradiente negativo della funzione obiettivo).
        
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
        
        # Inizializza vettore soluzione e residuo
        x = np.copy(x0)
        r = b - A.dot(x)  # Residuo iniziale r = b - Ax
        
        iterations = 0
        
        start_time = time.time()
        while iterations < max_iter:
            # Calcola Ar per determinare il passo ottimale
            Ar = A.dot(r)
            
            # Calcola il passo ottimale alpha usando la formula di minimizzazione
            # alpha = (r^T * r) / (r^T * A * r)
            alpha = (r @ r) / (r @ Ar)
            
            # Aggiorna la soluzione: x_new = x + alpha * r
            x_new = x + alpha * r
            
            # Verifica convergenza
            converged, rel_error = IterativeSolvers.convergence_check(A, x_new, b, tol)
            if converged:
                break
            
            # Aggiorna il residuo: r = b - A*x_new
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
        Utilizza direzioni coniugate per raggiungere la convergenza più rapidamente.
        
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
            
        nit = 0        # Contatore iterazioni
        err = 1        # Errore iniziale
        xold = x0      # Soluzione precedente
        rold = b - A @ xold  # Residuo iniziale r0 = b - Ax0
        pold = rold    # Prima direzione di ricerca = residuo iniziale
        
        start_time = time.time()
        
        while nit < max_iter and err > tol:
            # Calcola A*p per la formula del passo ottimale
            Ap = A @ pold
            
            # Calcola la lunghezza del passo nella direzione p
            # step = (p^T * r) / (p^T * A * p)
            step = (pold @ rold) / (pold @ Ap)
            
            # Aggiorna la soluzione: x_new = x_old + step * p
            xnew = xold + step * pold
            
            # Aggiorna il residuo: r_new = r_old - step * A * p
            rnew = rold - step * Ap
            
            # Calcola il fattore beta per la nuova direzione coniugata
            # beta = (A*p^T * r_new) / (A*p^T * p)
            beta = (Ap @ rnew) / (Ap @ pold)
            
            # Aggiorna la direzione coniugata: p_new = r_new - beta * p_old
            pnew = rnew - beta * pold
            
            # Calcola l'errore relativo
            err = np.linalg.norm(b - A @ xnew) / np.linalg.norm(b)
            
            # Aggiornamenti per la prossima iterazione
            xold = xnew
            rold = rnew
            pold = pnew
            nit += 1
            
        elapsed_time = time.time() - start_time
        
        if nit == max_iter:
            print(f"Il metodo del Gradiente Coniugato non ha raggiunto la convergenza in {max_iter} iterazioni.")
        
        return xnew, nit, elapsed_time, err
    
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
        # Dizionario che mappa i nomi dei metodi alle funzioni corrispondenti
        methods = {
            'jacobi': IterativeSolvers.jacobi,
            'gauss_seidel': IterativeSolvers.gauss_seidel,
            'gradient': IterativeSolvers.gradient_method,
            'conjugate_gradient': IterativeSolvers.conjugate_gradient
        }
        
        results = {}
        
        # Se richiesto di eseguire tutti i metodi
        if method == 'all':
            for name, func in methods.items():
                print(f"\nRisoluzione con metodo: {name}")
                # Esegue il metodo con i parametri forniti
                x_approx, iterations, elapsed_time, rel_err = func(A, b, tol)
                
                if x_approx is not None:
                    # Calcola l'errore relativo rispetto alla soluzione esatta
                    rel_error_solution = np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)
                    
                    # Salva i risultati nel dizionario
                    results[name] = {
                        'solution': x_approx,
                        'iterations': iterations,
                        'time': elapsed_time,
                        'residual_error': rel_err,
                        'solution_error': rel_error_solution
                    }
                    
                    # Stampa risultati
                    print(f"Errore relativo: {rel_error_solution}")
                    print(f"Iterazioni: {iterations}")
                    print(f"Tempo di calcolo: {elapsed_time:.6f} sec")
        else:
            # Se richiesto di eseguire un singolo metodo
            if method in methods:
                func = methods[method]
                # Esegue il metodo richiesto
                x_approx, iterations, elapsed_time, rel_err = func(A, b, tol)
                
                if x_approx is not None:
                    # Calcola l'errore relativo rispetto alla soluzione esatta
                    rel_error_solution = np.linalg.norm(x_approx - x_exact) / np.linalg.norm(x_exact)
                    
                    # Salva i risultati nel dizionario
                    results[method] = {
                        'solution': x_approx,
                        'iterations': iterations,
                        'time': elapsed_time,
                        'residual_error': rel_err,
                        'solution_error': rel_error_solution
                    }
                    
                    # Stampa risultati
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
        I file .mtx sono in formato Matrix Market, un formato standard per matrici sparse.
        
        Parametri:
        filename (str): Percorso del file .mtx
        
        Returns:
        scipy.sparse.csr_matrix: Matrice caricata dal file
        """
        try:
            # Legge la matrice dal file usando la funzione mmread di scipy
            matrix = mmread(filename)
            
            # Converti in formato CSR se necessario
            if isinstance(matrix, np.ndarray):
                return csr_matrix(matrix)  # Converte array denso in formato CSR
            else:
                return matrix.tocsr()  # Converte dalla rappresentazione sparse corrente a CSR
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
        
        # Per ogni file di matrice
        for mtx_file in matrix_files:
            print(f"\nCaricamento matrice: {mtx_file}")
            # Carica la matrice dal file
            A = IterativeSolvers.load_matrix_mtx(mtx_file)
            
            if A is None:
                continue
                
            n = A.shape[0]  # Dimensione della matrice
            
            # Step 1: Creare la soluzione esatta (vettore di tutti 1)
            # Questa è una scelta arbitraria ma comoda per i test
            x_exact = np.ones(n)
            
            # Step 2: Calcolare il vettore b = A*x_exact
            # In questo modo, sappiamo che la soluzione esatta del sistema Ax = b è x_exact
            b = A.dot(x_exact)
            
            matrix_results = {}
            
            # Per ogni tolleranza da testare
            for tol in tolerances:
                print(f"\nTest con tolleranza: {tol}")
                # Risolve il sistema con tutti i metodi e la tolleranza specificata
                matrix_results[tol] = IterativeSolvers.solve_system(A, b, x_exact, tol)
                
            # Salva i risultati per questa matrice
            results[mtx_file] = matrix_results
            
        return results