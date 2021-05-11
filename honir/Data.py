from os.path import normpath
import dask.dataframe as dd
import math
import random


def import_csv(path, dtype):
    df = dd.read_csv(path, dtype=dtype, assume_missing=True)
    df.compute()
    return df


def normalize_rating(n_stars: float) -> float:
    return (n_stars - 2.5) / 2.5


def sample_data_frame_count(df, n_samples):
    return df.sample(frac=(n_samples/len(df)))


def sample_data_frame_frac(df, frac):
    return df.sample(frac=frac)


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
        self.movies = sample_data_frame_frac(self.movies, 0.1)

        self.ratings = import_csv(dir + '/ratings.csv', {
            'userId': 'int64',
            'movieId': 'int64',
            'rating': 'float64',
            'timestamp': 'int64'
        })
        self.ratings = sample_data_frame_frac(self.ratings, 0.1)

        # 5 utilisateurs différents, 2 films différents
        self.tags = import_csv(dir + '/tags.csv', {
            'userId': 'int64',
            'movieId': 'int64',
            'tag': 'str',
            'timestamp': 'int64'
        })
        self.tags = sample_data_frame_frac(self.tags, 0.1)

        self.filter_data()

    def estimated_rating_for_tag_for_user(self, userId, tag):
        T, R, M = self.tags, self.ratings, self.movies

        movies_with_tag = dd.merge(
            M,
            T[T.tag == tag],
            on='movieId'
        )[M.columns].drop_duplicates()

        if len(movies_with_tag) == 0:
            return 0

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
        T = self.tags
        mean_rating_by_user_for_tag_pair = self.estimated_rating_for_tags_for_user_filter(
            userId, T.tag == tagA, T.tag == tagB)
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

    def utility_single_tag(self, userId, tag):
        T, R, M = self.tags, self.ratings, self.movies

        # coverage
        Ir = dd.merge(M, R[R.userId == userId], on='movieId').drop(
            columns=['userId', 'timestamp'])  # movies, rated by user

        lenIr = len(Ir)
        if lenIr == 0:
            return 0

        Irt = dd.merge(Ir, T[(T.userId == userId) & (T.tag == tag)]
                       ['movieId'].drop_duplicates(), on='movieId')  # movies, rated by user, tagged t by user

        lenIrt = len(Irt)
        if lenIrt == 0:
            return 0

        cov = min(lenIrt, lenIr - lenIrt) / lenIr

        # w_t = self.estimated_rating_for_tag_for_user(userId, tag)
        ratings = Irt['rating']
        w_t = normalize_rating(ratings).sum().compute() / len(Irt)

        # statistical significance
        sigma_t = Irt.rating.var().compute()
        if sigma_t == 0:
            return 0
        sig = min(2, abs(w_t) / (sigma_t / math.sqrt(lenIrt)))

        U = cov * sig * abs(w_t)
        print(tag, U)
        return U

    # 4. Identifying user preferences
    def selected_statements(self, userId):
        T, R, M = self.tags, self.ratings, self.movies
        parameters = {'min_items': 0, 'max_p_value': 0}  # \Theta

        Ir = dd.merge(M, R[R.userId == userId], on='movieId').drop(
            columns=['userId', 'timestamp'])  # movies, rated by user
        lenIr = len(Ir)
        TT = T[T.userId == userId]

        def f(tag):
            # movies, rated by user, tagged t by user
            Irt = dd.merge(Ir, TT[TT.tag == tag]
                           ['movieId'].drop_duplicates(), on='movieId')
            # movies, rated by user, tagged t by anyone
            # Irt = dd.merge(Ir, T[T.tag == tag]
            #                ['movieId'].drop_duplicates(), on='movieId')

            lenIrt = len(Irt)
            if lenIrt == 0:
                print(tag, 'lenIrt == 0')
                return 0

            cov = min(lenIrt, lenIr - lenIrt) / lenIr

            # w_t = self.estimated_rating_for_tag_for_user(userId, tag)
            ratings = Irt['rating']
            w_t = normalize_rating(ratings).sum().compute() / len(Irt)

            # statistical significance
            sigma_t = Irt.rating.var().compute()
            if sigma_t == 0:
                print(tag, 'sigma_t == 0')
                return 0
            sig = min(2, abs(w_t) / (sigma_t / math.sqrt(lenIrt)))

            U = cov * sig * abs(w_t)
            print(tag, f"{U=} {cov=} {sig=} {w_t=}")
            return U

        # 4.2 Generate candidate statements
        tags = T.tag.dropna().drop_duplicates()
        tags = tags.head(n=15)
        print(len(tags), 'tags')
        utility = tags.map(f)
        utility.name = "utility"
        C = dd.concat([tags, utility], axis=1).reset_index().nlargest(
            10, 'utility')

        return C

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
