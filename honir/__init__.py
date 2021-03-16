from honir.Data import Data


def main():
    d = Data()
    print(d.movies.head())
    print()
    print(d.ratings.head())
    print()
    print(d.tags.head())
    print()

    est_r = d.estimated_rating_for_tag_for_user(11, 'dark hero')
    print("-> estimated rating for tag 'dark hero' for user 11", est_r)
