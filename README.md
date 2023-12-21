# Progetto_Stratified_Shuffle_Split_23-24
### Corso di Programmazione e Metodi Sperimentali per l'Intelligenza Artificiale.
Componenti del gruppo:
* Andrea Scazzeri
* Edoardo Giacometto
* Mattia Muraro

Si intende implementare un toolbox che mediante una pipeline permetta all'utente di utilizzare il ***k-Nearest-Neighbor***
(kNN) specificando tutti i parametri. La pipeline includerà i seguenti step:
1. *Lettura dei dati* - la traccia richiede la lettura da file .csv ma sarebbe opportuno estendere l'implementazione
anche ad altri tipi di file (sfruttare il pattern strategy e valutare la combinazione con il factory)
2. *Pre-Processing*
    + Gestione dei dati mancanti
    + Divisione delle feature (variabili indipendenti e predizioni)
    + Feature scaling - data la possibilità di effettuare il feature scaling con due metodi si richiede l'implementazione
   del pattern strategy
3. *Splitting del dataset* - Le techiche di splitting sono: **Holdout** e **Stratified Shuffle Split**. Si richiede
l'implementazione del pattern factory insieme al pattern strategy per rendere il codice aperto alla nuova aggiunta di altri metodi di splitting
4. *kNN* vuole essere una classe che ricevendo in ingresso i data set di train e test fornisca la classificazione del test
5. *Calcolo delle metriche* - utilizzare il pattern strategy per calcolare le metriche
6. *Mostrare i risultati* (solo nel caso di uso di Stratified Shuffle Split) plottare tutte le metriche calcolate in 
funzione delle diverse divisioni del dataset in train e test set (boxplot)