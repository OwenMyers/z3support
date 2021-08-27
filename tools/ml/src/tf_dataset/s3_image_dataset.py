import tensorflow as tf
import psycopg2
import os

class ImageDataset(tf.data.Dataset):

    @staticmethod
    def get_num_samples(batch_size, data_set_size, train_percent, train):
        test_size = int(data_set_size * (100 - train_percent)/100)
        train_size = int(data_set_size * train_percent/100)

        if train:
            return int(train_size / batch_size)
        return int(test_size / batch_size)

    @classmethod
    def _generator(cls, num_samples, train, batch_size):
        # TODO have to set this up for shuffling
        lower_bound = 0
        upper_bound = batch_size
        if train:
            asc_or_desc = 'asc'
        else:
            asc_or_desc = 'desc'
        file_name_list = []
        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            print("Dataset generator loop")
            if len(file_name_list) == 0:
                print("query pulling ")
                try:
                    db_connection = psycopg2.connect(database='owen', user='owen', port='5432')
                    cursor = db_connection.cursor()
                    cursor.execute("""
                        select key from image_meta 
                        where dimension_x = 432 and dimension_y = 576
                        and rand_int >= {} and rand_int < {}
                        order by id {}
                    """.format(lower_bound, upper_bound, asc_or_desc))
                    file_name_list = cursor.fetchall()
                except (Exception, psycopg2.Error) as error:
                    print("Error while fetching data from PostgreSQL", error)
                finally:
                    if db_connection:
                        cursor.close()
                        db_connection.close()
                        print("PostgreSQL connection is closed")
                lower_bound = upper_bound
                upper_bound += batch_size

            cur_file_name = file_name_list.pop()[0]
            cur_file_and_path = os.path.join('/home/owen/s3-bucket', cur_file_name)
            cur_data = tf.io.read_file(cur_file_and_path)
            cur_file = tf.io.read_file(cur_file_and_path)
            image = tf.image.decode_jpeg(cur_file)
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.rgb_to_grayscale(image)
            yield (image, image)

    def __new__(cls, batch_size, train=True, train_percent=80):
        try:
            db_connection = psycopg2.connect( database='owen', user='owen', port= '5432' )
            cursor = db_connection.cursor()
            cursor.execute("select count(*) from image_meta where dimension_x = 432 and dimension_y = 576")
            data_set_size = cursor.fetchall()[0][0]
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if db_connection:
                cursor.close()
                db_connection.close()
                print("PostgreSQL connection is closed")
        num_samples = cls.get_num_samples(batch_size, data_set_size, train_percent, train)
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=(tf.TensorSpec(shape=(432, 576, 1), dtype=tf.float32), tf.TensorSpec(shape=(432, 576, 1), dtype=tf.float32)),
            args=(num_samples, train, batch_size)
        )