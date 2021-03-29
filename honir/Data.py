from os.path import normpath
import dask.dataframe as dd


def import_csv(path, dtype):
    df = dd.read_csv(path, dtype=dtype, assume_missing=True)
    df.compute()
    return df


def normalize_rating(n_stars: float) -> float:
    return (n_stars - 2.5) / 2.5


class Data:
    def __init__(self, dir='data/ml-20m'):
        print('loading…')
        # Nettoyage : 6.2

        # +2 tags
        self.movies = import_csv(dir + '/movies.csv', {
            'movieId': 'int64',
            'title': 'str',
            'genres': 'str'
        })
        print(len(self.movies), 'movies')

        self.ratings = import_csv(dir + '/ratings.csv', {
            'userId': 'int64',
            'movieId': 'int64',
            'rating': 'float64',
            'timestamp': 'int64'
        })
        print(len(self.ratings), 'ratings')

        # 5 utilisateurs différents, 2 films différents
        self.tags = import_csv(dir + '/tags.csv', {
            'userId': 'int64',
            'movieId': 'int64',
            'tag': 'str',
            'timestamp': 'int64'
        })
        print(len(self.tags), 'tags')
        print()
        print()

    def estimated_rating_for_tag_for_user(self, userId, tag):
        T, R, M = self.tags, self.ratings, self.movies

        movies_with_tag = dd.merge(
            M,
            T[T.tag == tag],
            on='movieId'
        )[M.columns].drop_duplicates()

        movies_rated_by_user = dd.merge(
            movies_with_tag,
            R[R.userId == userId],
            on='movieId'
        )

        ratings = movies_rated_by_user['rating']
        normalized_ratings = normalize_rating(ratings)

        mean_rating_by_user_for_tag = (
            normalized_ratings.sum().compute() / len(movies_with_tag)
        )

        return mean_rating_by_user_for_tag

    def estimated_rating_for_tags_for_user(self, userId, tagA, tagB):
        T, R, M = self.tags, self.ratings, self.movies

        ids_movies_with_both_tags = dd.merge(
            T[T.tag == tagA],
            T[T.tag == tagB],
            on="movieId"
        )[('movieId')]

        ratings = dd.merge(
            R[R.userId == userId],
            ids_movies_with_both_tags,
            on='movieId'
        )['rating']

        normalized_ratings = normalize_rating(ratings)

        mean_rating_by_user_for_tag_pair = (
            normalized_ratings.sum().compute() / len(ids_movies_with_both_tags)
        )

        return mean_rating_by_user_for_tag_pair

    def estimated_rating_for_tags_for_user_filter(self, userId, f1, f2):
        T, R, M = self.tags, self.ratings, self.movies

        ids_movies_with_both_tags = dd.merge(
            T[f1],
            T[f2],
            on="movieId"
        )[('movieId')]

        ratings = dd.merge(
            R[R.userId == userId],
            ids_movies_with_both_tags,
            on='movieId'
        )['rating']

        normalized_ratings = normalize_rating(ratings)

        mean_rating_by_user_for_tag_pair = (
            normalized_ratings.sum().compute() / len(ids_movies_with_both_tags)
        )

        return mean_rating_by_user_for_tag_pair
        
        
    def filter_data(self):
        print('filtering on tag vocabulary...')
        print()
        print()
        
        T = self.tags

        nb_users = T[["userId", "tag"]].drop_duplicates().tag.value_counts().compute()
        
        # only keep tags used by more than 5 different users
        nb_users = nb_users.where(lambda x : x>=5).dropna()
        T_5 = T.loc[T['tag'].isin(nb_users.keys())]
        
        nb_movies = T_5[["movieId", "tag"]].drop_duplicates().tag.value_counts().compute()
        
        # only keep tags used on more than 2 different movies
        nb_movies = nb_movies.where(lambda x : x>=2).dropna()
        T_2 = T_5.loc[T['tag'].isin(nb_movies.keys())]
        
        return T_2
