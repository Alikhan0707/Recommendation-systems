import pandas as pd
from scipy.sparse import csr_matrix

def prefilter_items(data_train, item_feat, take_n_popular=5000):
    
    weeks_in_year = 52
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    data_train_less_n_month = data_train.loc[data_train['week_no'] >= data_train['week_no'].max() - weeks_in_year]
    
    popularity = data_train_less_n_month.groupby('item_id')[['quantity', 'sales_value']].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold', 'sales_value': 'price'}, inplace=True)
    popularity = popularity.loc[(popularity.n_sold > 0) & (popularity.price > 0)]
    popularity['new_sales_value'] = popularity.price / popularity.n_sold
    
    popularity = popularity.sort_values('new_sales_value', ascending=False).reset_index()
    
    # Уберем слишком дорогие товар
    
    popularity = popularity.loc[3::]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    
    popularity = popularity.loc[popularity['new_sales_value'] > 1] 
    
    # Уберем самые непопулряные
    popularity = popularity.loc[popularity['n_sold'] > 50]
    
    # Уберем самые популярные (топ-3)
     
    popularity = popularity.sort_values('n_sold', ascending=False).reset_index()
    popularity = popularity.loc[3::]
   
    # Уберем не интересные для рекоммендаций категории (department)
    small_count_departments = ['VIDEO RENTAL', 'AUTOMOTIVE', 'HOUSEWARES', 
            'PORK', 'POSTAL CENTER', 'GM MERCH EXP', 'CNTRL/STORE SUP', 
            'PROD-WHS SALES', 'DAIRY DELI', 'HBC', 'CHARITABLE', 'RX', 'TOYS',
            'PHOTO', 'DELI/SNACK BAR', 'GRO BAKERY', 'PHARMACY SUPPLY', 
            'ELECT &PLUMBING', 'MEAT-WHSE', 'VIDEO']
    
    
    item_feat_ids = item_feat.loc[~item_feat['department'].isin(small_count_departments), 'item_id']
    
    popularity = popularity.loc[popularity.item_id.isin(item_feat_ids), 'item_id'].tolist()
    
    top_items = popularity[:take_n_popular]
    
    # Добавим, чтобы не потерять юзеров
    data_train.loc[~data_train['item_id'].isin(top_items), 'item_id'] = 999999 

    return data_train


def postfilter_items():
    pass


def train_test_split_by_week(df, name_of_week_column, test_size_weeks):

    week_ratio = df[name_of_week_column].max() - test_size_weeks

    df_train = df[df[name_of_week_column] < week_ratio]
    df_test = df[df[name_of_week_column] >= week_ratio]

    return df_train, df_test


def group_df_by(df, grouped_feat, second_feat, new_name_of_second_feat, pick_method):

    if pick_method == 'unique':
        grouped_df = df.groupby(grouped_feat)[second_feat].unique().reset_index()
        grouped_df.columns = [grouped_feat, new_name_of_second_feat]

    elif pick_method == 'sum':
        grouped_df = df.groupby(grouped_feat)[second_feat].sum().reset_index()
        grouped_df.columns = [grouped_feat, new_name_of_second_feat]

    return grouped_df


def df_to_user_item_matrix(df):

    user_item_matrix = pd.pivot_table(df,
                                      index='user_id',
                                      columns='item_id',
                                      values='quantity',
                                      aggfunc='count',
                                      fill_value=0)

    user_item_matrix = user_item_matrix.astype(float)

    sparce_user_item = csr_matrix(user_item_matrix).tocsr()

    sparce_t_user_item = csr_matrix(user_item_matrix).T.tocsr()

    return user_item_matrix, sparce_user_item, sparce_t_user_item

def group_df_train_and_test(train, test, groupby_feat, second_feat, rename_column, aggfunc):
    
    assert aggfunc in ['unique', 'sum'], 'aggfunc принимает значения unique, sum'
    
    train_df = group_df_by(train, groupby_feat, second_feat, rename_column, aggfunc)
    test_df = group_df_by(test, groupby_feat, second_feat, rename_column, aggfunc)
        
    return train_df, test_df

def change_ids(matrix):
    user_id = matrix.index.values
    item_id = matrix.columns.values

    matrix_user_id = np.arange(len(user_id))
    matrix_item_id = np.arange(len(item_id))

    id_to_user_id = dict(zip(matrix_user_id, user_id))
    id_to_item_id = dict(zip(matrix_item_id, item_id))

    user_id_to_id = dict(zip(user_id, matrix_user_id))
    item_id_to_id = dict(zip(item_id, matrix_item_id))

    return id_to_user_id, id_to_item_id, item_id_to_id, user_id_to_id


def get_recommendation(user, model, sparce_user_item, id_to_item_id, user_id_to_id, N=5):

    res = [id_to_item_id[rec[0]] for rec in model.recommend(userid=user_id_to_id[user],
                                                            user_items=sparce_user_item,
                                                            N=N,
                                                            filter_already_liked_items=False,
                                                            filter_items=None,
                                                            recalculate_user=True)]

    return res



