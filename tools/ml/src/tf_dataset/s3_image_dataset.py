import tensorflow as tf
import psycopg2

class ImageDataset(tf.data.Dataset):
    def manual_init(self, batch_sizes):
        self.manual_initialized = True
        self.data_set_size = None
        try:
            db_connection = psycopg2.connect( database='owen', user='owen', port= '5432' )
            cursor = db_connection.cursor()
            cursor.execute("select count(*) from image_meta where dimension_x = 432 and dimension_y = 576")
            self.data_set_size = cursor.fetchall()[0][0]
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if db_connection:
                cursor.close()
                db_connection.close()
                print("PostgreSQL connection is closed")
                raise Exception('Unexpected SQL Error')

        self.batch_sizes = batch_sizes

    @staticmethod
    def get_num_samples(batch_sizes, data_set_size, percent_test_set):
        test_size = int(data_set_size * percent_test_set)
        train_size = data_set_size - test_size
        num_samples = int(data_set_size/batch_sizes)

        return num_samples

    def _generator(self):
        num_samples = self.get_num_samples(self.batch_sizes, self.data_set_size, 0.0)
        try:
            db_connection = psycopg2.connect( database='owen', user='owen', port= '5432' )
            cursor = db_connection.cursor()
            cursor.execute("""
                select key from image_meta 
                where dimension_x = 432 and dimension_y = 576
                order by id
            """)
            self.data_set_size = cursor.fetchall()[0][0]
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if db_connection:
                cursor.close()
                db_connection.close()
                print("PostgreSQL connection is closed")
                raise Exception('Unexpected SQL Error')

        for sample_idx in range(num_samples):
            # Reading data (line, record) from the file
            print("Dataset generator loop")
            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_shapes =  (1,),
            output_types = tf.int64,
            args=(num_samples,)
        )