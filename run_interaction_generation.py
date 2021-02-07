from os.path import join
import time
import pickle
from cric import CRIC, PriorityQueue, load_cric
from loaddata import load_data
from preprocess import generate_interaction

if __name__ == "__main__":
    dataset = "criteo"
    split_type_folder = join("./data", dataset, "random_split")
    train_valid_test_id = [list(range(3, 11)), [], []]
    train_generator, _, _, target, dense_features, sparse_features, _ = \
        load_data(dataset, split_type_folder, train_valid_test_id)
    columns = target + dense_features + sparse_features

    time_path = join(split_type_folder, "running_time.pkl")
    if os.path.exists(time_path):
        with open(time_path, "rb") as handle:
            running_time = pickle.load(handle)
    else:
        running_time = {0: 0}

    cric = CRIC(200, 50, max_size=5, n_chain=100, positive_class=True)
    for epoch in range(1, 151):
        print("epoch", epoch)
        running_time[epoch] = running_time[epoch - 1]
        for file_count, (X_train, y_train) in enumerate(train_generator()):
            time_start = time.time()
            cric.fit(X_train[sparse_features], y_train, [0])
            running_time[epoch] += time.time() - time_start
        print("runnning time:", running_time[epoch])
        print("number of interactive features:", len(cric.new_features))
        cric.save(join(split_type_folder, "crics", "cric_{}.pkl".format(epoch)))
        with open(time_path, "wb") as handle:
            pickle.dump(running_time, handle, protocol=pickle.HIGHEST_PROTOCOL)

    cric = load_cric(join(split_type_folder, "crics", "cric_{}.pkl".format(150)))
    for i in range(10):
        print("now part %d" % (i + 1))
        part_folder = join(split_type_folder, "part{}".format(i + 1))
        generate_interaction(part_folder, "train.npy", columns, cric)
    print(len(cric.new_features))