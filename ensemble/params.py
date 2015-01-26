# 用于搜索参数空间，类似于grid_search的作用

import pandas as pd

np.random.seed(123)

df = pd.DataFrame(columns=('w1', 'w2', 'w3', 'mean', 'std'))

i = 0
for w1 in range(1,4):
    for w2 in range(1,4):
        for w3 in range(1,4):
            
            if len(set((w1,w2,w3))) == 1: # skip if all weights are equal
                continue
            
            eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[w1,w2,w3])
            scores = cross_validation.cross_val_score(
                                            estimator=eclf,
                                            X=X, 
                                            y=y, 
                                            cv=5, 
                                            scoring='accuracy',
                                            n_jobs=1)
            
            df.loc[i] = [w1, w2, w3, scores.mean(), scores.std()]
            i += 1
            
df.sort(columns=['mean', 'std'], ascending=False)