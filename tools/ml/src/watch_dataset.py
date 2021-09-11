# The generator version of a dataset can be harder to quickly check/investigate
# Than other dataset forms. This tool makes it easy to look at what the dataset
# is passing in to the model

from tf_dataset.minst_dataset import MnistDataset
import time


def benchmark(dataset, num_epochs=2):
    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        for sample in dataset:
            # Performing a training step
            time.sleep(0.01)
    print("Execution time:", time.perf_counter() - start_time)


def show_data(dataset):

    print("Trying to plot dataset")

def main():
    benchmark(MnistDataset(batch_size=10).range(100))
    show_data()

if __name__ == "__main__":
    main()