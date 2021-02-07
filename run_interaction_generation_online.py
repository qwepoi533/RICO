from os.path import join
from cric import CRIC, PriorityQueue, load_cric
from loaddata import load_data
from preprocess import generate_interaction


if __name__ == "__main__":
    decay = 1
    dataset = "avazu"
    split_type_folder = join("./data", dataset, "ordered_split")
    cric_name = "cric.pkl" if decay == 1 else "cric_decay{}.pkl".format(decay)
    train_valid_test_id = [list(range(1, 9)), [], []]
    train_generator, _, _, target, dense_features, sparse_features, _ = \
        load_data(dataset, split_type_folder, train_valid_test_id)
    columns = target + dense_features + sparse_features
    cric = CRIC(200, 50, max_size=5, n_chain=10000, decay=decay, online=True, positive_class=True)
    for X_train, y_train in train_generator():
        cric.fit(X_train[sparse_features], y_train, [0])
    print("number of interactive features:", len(cric.new_features))
    cric.save(join(split_type_folder, cric_name))

    cric = load_cric(join(split_type_folder, cric_name))
    for i in range(4, 11):
        print("now part %d" % i)
        part_folder = join(split_type_folder, "part{}".format(i))
        generate_interaction(part_folder, "train.npy", columns, cric)
