import pickle
import os


def main():
    models_path = 'models/'
    model_dir_list = os.listdir(models_path)
    for cur_f in model_dir_list:
        if ".pkl" not in cur_f:
            continue
        with open(os.path.join(models_path, cur_f), "rb") as f:
            info_dict = pickle.load(f)
        print(info_dict)


if __name__ == "__main__":
    main()