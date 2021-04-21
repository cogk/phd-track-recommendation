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

        # only keep tags used by more than 5 different users
        tags_users = T[['tag', 'userId']
                       ].drop_duplicates().tag.value_counts().compute()
        tags_users = tags_users.where(lambda x: x >= 5).dropna()
        T_tags_users = T.loc[T['tag'].isin(tags_users.keys())]

        print('tags_users')
        print(tags_users.head())
        print()
        print()

        # only keep tags used on more than 2 different movies
        tags_movies = T_tags_users[['tag', 'movieId']
                                   ].drop_duplicates().tag.value_counts().compute()
        tags_movies = tags_movies.where(lambda x: x >= 2).dropna()
        T_tags_movies = T_tags_users.loc[T_tags_users['tag'].isin(
            tags_movies.keys())]

        print('tags_movies')
        print(tags_movies.head())
        print()
        print()

        # only keep movies that have at least 2 tags
        movies_tags = T_tags_movies[['movieId', 'tag']].drop_duplicates(
        ).movieId.value_counts().compute()
        movies_tags = movies_tags.where(lambda x: x >= 2).dropna()
        T_movies_tags = T_tags_movies.loc[T_tags_movies['movieId'].isin(
            movies_tags.keys())]

        print('movies_tags')
        print(movies_tags.head())
        print(movies_tags.tail())
        print()
        print()

        # for each movie, only keep tags that have been assigned by at least 2 users
        movies_tags_users = T_movies_tags[['movieId', 'tag', 'userId']] .drop_duplicates(
        ).groupby(['movieId', 'tag']).size().compute()
        movies_tags_users = movies_tags_users.where(lambda x: x >= 2).dropna()
        boolResult = T_movies_tags[['movieId', 'tag']].apply(
            tuple, 1, meta=('object')).isin(movies_tags_users.keys())
        T_movies_tags_users = T_movies_tags.loc[boolResult]

        return T_movies_tags_users
