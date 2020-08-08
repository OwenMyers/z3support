import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    args = parser.parse_args()

    if not os.path.exists(args.src):
        raise ValueError('Invalid src destination')


if __name__ == '__main__':
    main()

