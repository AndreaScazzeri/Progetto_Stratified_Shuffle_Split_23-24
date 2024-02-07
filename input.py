import os
class Input_parametriEsecuzione():
    @staticmethod
    def parametriEsecuzione():
        while True:
            path = input("Inserisci il Path: ")
            if os.path.exists(path):
                break
            else:
                print("Inserire un Path valido!")
        while True:
            fs = input(
                "\nInserisci il tipo di Feature Scaling, scrivere 'stand' (default) se si vuole la standardizzazione, scrivere 'norm' se si vuole la normalizazione. \n "
                "Non inserire niente se si vuole scegliere il Feature Scaling di default: ")
            if fs.lower() == "stand" or fs.lower() == "norm":
                fs = fs.lower()
                break
            elif fs.lower() == "":
                fs = "stand"
                break
            else:
                print("Input inserito errato, scrivere 'stand' o 'norm'!")

        while True:
            splitting_type = input("\nInserisci il metodo di Splitting Type, 'holdout' (default) se si vuole eseguire l'holdout,\n"
                                   " 'sss' se si vuole eseguire lo Stratified Shuffle Split opure 'both' per selezionare entrambi. \n"
                                   " Non inserire niente se si vuole selezionare il metodo di default: ")
            if splitting_type.lower() == "holdout" or splitting_type.lower() == "sss" or splitting_type.lower() == "both":
                splitting_type = splitting_type.lower()
                break
            elif splitting_type.lower() == "":
                splitting_type = "holdout"
                break
            else:
                print("Input inserito errato, scrivere 'holdout' o 'sss' o 'both'!")

        while True:
            parametro_splitting = input(
                "\nInserisci il Parametro di Splitting, é la percentuale di data set che entra nel test set.\n"
                " Non inserire niente se si vuole scegliere il numero di default (0.2): ")
            if parametro_splitting == "":
                parametro_splitting = 0.2
                break
            elif float(parametro_splitting) and float(parametro_splitting) < 1 and float(parametro_splitting) > 0 :
                parametro_splitting = float(parametro_splitting)
                break
            else:
                print("Inserire un numero valido che sia un float maggiore di 0 ma minore di 1!")
        while True:
            n_divisioni = input("\nInserire il numero di Esperimenti per lo Stratified Shuffle Split.\n"
                                "Non inserire niente se si vuole scegliere il numero di default (5): ")
            if n_divisioni == "":
                n_divisioni = 5
                break
            elif int(n_divisioni) > 0:
                n_divisioni = int(n_divisioni)
                break
            else:
                print("Inserire un numero di divisioni maggiore di 0!")
        while True:
            k = input("\nInserire il parametro K per il KNN. Non inserire niente se si vuole scegliere il K di default (7): ")
            if k == "":
                k = 7
                break
            elif int(k) > 0 and int(k) % 2 != 0:
                k = int(k)
                break
            else:
                print("Inserire un valore K che sia maggiore di 0 e dispari!")
        while True:
            ar = input("\nDigitare 'True' se si vuol veririficare l'Accuracy Rate, 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if ar.lower() == "true":
                ar = True
                break
            elif ar == "" or ar.lower() == "false":
                ar = False
                break
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            er = input("\nDigitare 'True' se si vuol veririficare l'Error Rate, 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if er.lower() == "true":
                er = er.lower().capitalize()
                break
            elif er == "" or er.lower() == "false":
                er =  False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            sens = input("\nDigitare 'True' se si vuol veririficare l'Sensitivity Rate, 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if sens.lower() == "true":
                sens = True
                break
            elif sens == "" or sens.lower() == "false":
                sens = False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            spec = input("\nDigitare 'True' se si vuol veririficare l'Specificity Rate, 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if spec.lower() == "true":
                spec = True
                break
            elif spec == "" or spec.lower() == "false":
                spec = False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            gm = input("\nDigitare 'True' se si vuol veririficare l'Geometry Mean, 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if gm.lower() == "true":
                gm =  True
                break
            elif gm == "" or gm.lower() == "false":
                gm = False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            all_metrics = input("\nDigitare 'True'(default) se si vogliono verificare tutte le metriche precedenti, 'False'  altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if all_metrics.lower() == "false":
                all_metrics = False
                break
            elif all_metrics == "" or all_metrics.lower() == "true":
                all_metrics = True
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            seed = input("\nDigitare il Seed, non inserire niente se non lo si vuole specificare: ")
            if seed == '':
                seed = None
                break
            elif int(seed) >= 0:
                seed = int(seed)
                break
            else:
                print("Inserire un valore intero o nullo se non si vuole specificare il Seed")
        while True:
            show_boxplot = input("\nDigitare 'True' se si vuol visualizzare il Box Plot (visualizzabile solo se si é scelto lo stratified), 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if show_boxplot.lower() == "true":
                show_boxplot = True
                break
            elif show_boxplot == '' or show_boxplot.lower() == "false":
                show_boxplot = False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            show_lineplot = input("\nDigitare 'True' se si vuol visualizzare il Line Plot (visualizzabile solo se si é scelto lo stratified), 'False' (default) altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if show_lineplot.lower() == "true":
                show_lineplot = True
                break
            elif show_lineplot == '' or show_lineplot.lower() == "false":
                show_lineplot = False
                break
            else:
                print("Scrivere 'True' o 'False'!")
        while True:
            show_table = input("\nDigitare 'True' (default) se si vuol visualizzare la tabella delle performance, 'False'  altrimenti.\n"
                       " Non inserire niente per scegliere l'opzione di default: ")
            if show_table.lower() == "false":
                show_table = False
                break
            elif show_table == '' or show_table.lower() == "true":
                show_table = True
                break
            else:
                print("Scrivere 'True' o 'False'!")
        return(
        path, fs, splitting_type, parametro_splitting, n_divisioni, k, ar, er, sens, spec, gm, all_metrics, seed,
        show_boxplot, show_lineplot, show_table)
