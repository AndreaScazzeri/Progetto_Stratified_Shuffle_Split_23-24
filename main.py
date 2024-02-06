from kNNPipeline import KNNPipeline
from Input import Input_parametriEsecuzione
if __name__ == '__main__':
    input = Input_parametriEsecuzione.parametriEsecuzione()
    pipeline = KNNPipeline(input[0], fs=input[1], splitting_type= input[2], parametro_splitting=input[3],
                 n_divisioni=input[4], k= input[5], ar = input[6], er = input[7], sens = input[8], spec = input[9], gm= input[10],
                 all_metrics= input[11], seed= input[12], show_boxplot= input[13], show_lineplot= input[14], show_table=input[15])
    print('Progetto Stratified Shuffle')
