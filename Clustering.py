import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SOM:
    def __init__(this, width, height, input_dim):
        this.width = width
        this.height = height
        this.input_dim = input_dim
        this.weight = tf.Variable(tf.random_normal([this.width * this.height, this.input_dim]))
        this.input = tf.placeholder(tf.float32, [this.input_dim])
        this.location = tf.to_float([[y,x] for y in range(height) for x in range(width)])
        this.bmu = this.get_bmu()
        this.update_weight = this.update_neighbour()

    def get_bmu(this):
        square_diff = tf.square(this.input - this.weight)
        distance = tf.sqrt(tf.reduce_mean(square_diff, axis = 1))
        bmu_index = tf.argmin(distance)
        bmu_loc = tf.to_float([tf.div(bmu_index, this.width), tf.mod(bmu_index, this.width)])
        return bmu_loc

    def update_neighbour(this):
        lr = .1
        sigma = tf.to_float(tf.maximum(this.height, this.width) / 2)
        square_diff = tf.square(this.bmu - this.location)
        distance = tf.sqrt(tf.reduce_mean(square_diff, axis = 1))
        neighbour_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), tf.square(sigma) * 2))
        rate = neighbour_strength * lr
        rate_stack = tf.stack([tf.tile(tf.slice(rate, [i], [1]), [this.input_dim]) for i in range(this.width * this.height)])
        input_weight_diff = this.input - this.weight
        weight_diff = rate_stack * input_weight_diff
        updated_weight = this.weight + weight_diff
        return tf.assign(this.weight, updated_weight)
    
    def train(this, dataset, epoch):
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(epoch):
                for data in dataset:
                    sess.run(this.update_weight, feed_dict = {this.input: data})
            location = sess.run(this.location)
            weight = sess.run(this.weight)
            print(weight)
            clusters = [[] for i in range(this.height)]
            for i, loc in enumerate(location):
                clusters[int(loc[0])].append(weight[i])
            this.clusters = clusters

def main():
    dataset = pd.read_csv("credit_card_general_clustering.csv")
    dataset = dataset[["BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "PURCHASES_FREQUENCY", 
                "ONEOFF_PURCHASES_FREQUENCY", "PURCHASES_TRX", "PAYMENTS", "MINIMUM_PAYMENTS", 
                "PRC_FULL_PAYMENT", "TENURE"]]
    dataset["MINIMUM_PAYMENTS"].fillna(dataset["MINIMUM_PAYMENTS"].mean(), inplace=True)

    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)

    pca = PCA(n_components = 3)
    pca_component = pca.fit_transform(dataset)

    width = 6
    height = 6
    input_dim = 3
    epoch = 5000

    som = SOM(width, height, input_dim)
    som.train(pca_component, epoch)
    plt.imshow(som.clusters)
    plt.show()

main()