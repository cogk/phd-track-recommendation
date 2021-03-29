from honir.Data import Data


def main():
    d = Data()
    T, R, M = d.tags, d.ratings, d.movies
    # print(d.movies.head())
    # print()
    # print(d.ratings.head())
    # print()
    # print(d.tags.head())
    # print()

    est_r_ta = d.estimated_rating_for_tag_for_user(11, 'dark hero')
    print("-> weight('dark hero', user=11)", est_r_ta)

    est_r_ta_connective_tb = d.estimated_rating_for_tags_for_user(
        11, 'dark hero', 'surreal')
    w_ta_connective_tb = est_r_ta_connective_tb - est_r_ta
    print("-> weight('dark hero' & 'surreal', user=11)", w_ta_connective_tb)

    est_r_ta_connective_not_tb = d.estimated_rating_for_tags_for_user_filter(
        11, T.tag == 'dark hero', T.tag != 'surreal')
    w_ta_connective_not_tb = est_r_ta_connective_not_tb - est_r_ta
    print("-> weight('dark hero' but not 'surreal', user=11)",
          w_ta_connective_not_tb)
    
    print()
    filteredTags = d.filter_data()
    print(filteredTags.head())
