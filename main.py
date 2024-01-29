from kNNPipeline import KNNPipeline
if __name__ == '__main__':
    pipeline = KNNPipeline('Progetto_Stratified_Shuffle_Split_23-24/breast_cancer_test.csv',splitting_type='sss',fs='norm',parametro_splitting=3)
    print('Progetto Stratified Shuffle')