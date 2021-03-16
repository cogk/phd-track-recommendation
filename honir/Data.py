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
