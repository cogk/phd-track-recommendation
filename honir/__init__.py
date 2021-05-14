from honir.Data import Data


def main():
    d = Data()

    T, R, M = d.tags, d.ratings, d.movies

    print(len(M), 'movies')
    print(len(R), 'ratings')
    print(len(T), 'tags')
    print()

    # userId = 70201
    userId = R['userId'].head(1).values[0]

    est_r_ta = d.estimated_rating_for_tag_for_user(userId, 'dark hero')
    print(f"-> weight('dark hero', {userId=})", est_r_ta)

    est_r_ta_connective_tb = d.estimated_rating_for_tags_for_user(
        userId, 'dark hero', 'surreal')
    w_ta_connective_tb = est_r_ta_connective_tb - est_r_ta
    print(f"-> weight('dark hero' & 'surreal', {userId=})", w_ta_connective_tb)

    est_r_ta_connective_not_tb = d.estimated_rating_for_tags_for_user_filter(
        userId, T.tag == 'dark hero', T.tag != 'surreal')
    w_ta_connective_not_tb = est_r_ta_connective_not_tb - est_r_ta
    print(f"-> weight('dark hero' but not 'surreal', {userId=})",
          w_ta_connective_not_tb)

    print()
    print('.selected_statements(userId=' + str(userId) + ')')
    C = d.selected_statements(userId)
    print(C.head())
    
    print()
    filteredTags = d.filter_data()
    print(filteredTags.head())
