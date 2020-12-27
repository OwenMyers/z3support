import datetime
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Make a blank time stamped settings file.')
    parser.add_argument('-f', type=str, help='File location', default='./')
    args = parser.parse_args()

    now = datetime.datetime.now()

    datetime_str = now.isoformat()

    with open(os.path.join(args.f, datetime_str + '.conf'), 'w') as f:
        f.write('[Settings]\n')
        f.write(f'timestamp={datetime_str}')


if __name__ == '__main__':
    main()
