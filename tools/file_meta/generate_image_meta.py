import sqlalchemy
import numpy as np
from sqlalchemy.orm import sessionmaker
import os
import boto3
from PIL import Image
from io import BytesIO
import model

engine = sqlalchemy.create_engine(os.getenv("POSTGRES_DSN"))

WASABI_ACCESS_KEY = os.getenv("WASABI_ACCESS_KEY")
WASABI_SECRET_KEY = os.getenv("WASABI_SECRET_KEY")


def main():
    Session = sessionmaker(bind=engine)
    session = Session()


    s3 = boto3.resource(
        's3',
        endpoint_url='https://s3.wasabisys.com',
        aws_access_key_id=WASABI_ACCESS_KEY,
        aws_secret_access_key=WASABI_SECRET_KEY
    )
    bucket_name = 'zillow-images'
    bucket = s3.Bucket(bucket_name)
    for i, object in enumerate(bucket.objects.all()):
        print(object)
        print(i)
        # from https://stackoverflow.com/questions/44043036/how-to-read-image-file-from-s3-bucket-directly-into-memory
        response = object.get()
        file_stream = response['Body']
        im = Image.open(file_stream)
        a = np.array(im)
        if len(a.shape) < 2:
            continue
        elif len(a.shape) < 3:
            x = int(a.shape[0])
            y = int(a.shape[1])
            z = None
        else:
            x = int(a.shape[0])
            y = int(a.shape[1])
            z = int(a.shape[2])
        image_meta = model.ImageMeta(
            x,
            y,
            z,
            bucket_name,
            object.key,
            int(a.max()),
            int(a.min()),
            int(a.std()),
        )

        session.add(image_meta)
        session.commit()


if __name__ == "__main__":
    main()
