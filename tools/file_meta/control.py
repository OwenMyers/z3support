import argparse
import model
import generate_image_meta
import sqlalchemy
import os


engine = sqlalchemy.create_engine(os.getenv("POSTGRES_DSN"))
parser = argparse.ArgumentParser()


def main():

    parser.add_argument('-u', action='store_true', dest='set_up', help='Create database')
    #parser.add_argument('-d', action='store_true', dest='clear_all', help='Drop everything')

    args = parser.parse_args()

    if args.set_up:
        model.Base.metadata.create_all(generate_image_meta.engine)

if __name__ == "__main__":
    main()