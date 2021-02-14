import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

from sklearn.metrics.pairwise import cosine_similarity

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from src.utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, items_feat, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.items_features = items_feat
        
        self.user_item_matrix = self.prepare_matrix(data, self.items_features)  # pd.DataFrame
        
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = \
                self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data, items_feat):
        
        data = prefilter_items(data, items_feat)
        
        user_item_matrix = pd.pivot_table(data, 
                                          index='user_id', columns='item_id', 
                                          values='quantity', 
                                          aggfunc='count', 
                                          fill_value=0)
        
        user_item_matrix = user_item_matrix.astype(float)
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
    
    def get_recommendations(self, user, model, sparse_user_item, N=5):
        """Рекомендуем топ-N товаров"""
        res = [self.id_to_itemid[rec[0]] for rec in 
                        model.recommend(userid=self.userid_to_id[user], 
                                        user_items=sparse_user_item,   # на вход user-item matrix
                                        N=N, 
                                        filter_already_liked_items=False, 
                                        filter_items=[self.itemid_to_id[999999]],  # !!! 
                                        recalculate_user=True)]
        return res
    
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        own_recommender = self.fit_own_recommender(self.user_item_matrix)
        
        sparce_user_item = csr_matrix(self.user_item_matrix).tocsr()
        
        res = self.get_recommendations(user, own_recommender, sparce_user_item, N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
    
    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        
        own_recommender = self.fit_own_recommender(self.user_item_matrix.T)
        
        sparce_user_item = csr_matrix(self.user_item_matrix).tocsr()
        
        res = self.get_recommendations(user, own_recommender, sparce_user_item, N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
