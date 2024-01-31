from kNNPipeline import KNNPipeline
if __name__ == '__main__':
    pipeline = KNNPipeline('breast_cancer_test.csv', splitting_type='sss',n_divisioni=3)
    print('Progetto Stratified Shuffle')
