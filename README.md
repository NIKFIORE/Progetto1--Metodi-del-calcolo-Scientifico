# Progetto1--Metodi-del-calcolo-Scientifico
### **Descrizione del Progetto - Mini Libreria per Sistemi Lineari**  

Il progetto richiede l'implementazione di una mini libreria per la risoluzione di sistemi lineari utilizzando metodi iterativi per matrici **simmetriche e definite positive**.  

#### **Metodi da implementare:**  
1. Metodo di **Jacobi**  
2. Metodo di **Gauß-Seidel**  
3. Metodo del **Gradiente**  
4. Metodo del **Gradiente Coniugato**  

#### **Requisiti tecnici:**  
- Utilizzare un linguaggio di programmazione a scelta (C++, Fortran, Java, Python, ecc.).  
- Partire da una libreria open-source (Eigen, Armadillo, BLAS/LAPACK) solo per la gestione di vettori e matrici, **senza usare metodi preimplementati** per la risoluzione dei sistemi.  
- I metodi iterativi devono partire da un vettore iniziale nullo e arrestarsi quando:  
  - L'errore relativo \( ||Ax^{(k)} - b|| / ||b|| \) è inferiore a una tolleranza \( \text{tol} \).  
  - Il numero massimo di iterazioni (minimo **20000**) è raggiunto.  

#### **Formato dell'eseguibile:**  
L'eseguibile dovrà accettare in input:  
- Una **matrice** \( A \) (simmetrica e definita positiva).  
- Un **vettore** \( b \) (termine noto).  
- Un **vettore soluzione esatta** \( x \).  
- Una **tolleranza** \( \text{tol} \).  

Dovrà quindi eseguire tutti i metodi e restituire:  
- **Errore relativo** tra la soluzione esatta e quella calcolata.  
- **Numero di iterazioni richieste**.  
- **Tempo di calcolo**.  

#### **Validazione e test:**  
- Applicare il codice alle **matrici sparse** fornite nei file `.mtx` (spa1.mtx, spa2.mtx, vem1.mtx, vem2.mtx).  
- Eseguire test con diverse tolleranze:  
  - \( 10^{-4}, 10^{-6}, 10^{-8}, 10^{-10} \).  
