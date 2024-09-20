import numpy as np
import tensorflow as tf
from model import create_model

tf.config.set_visible_devices([], 'GPU')

class Client:
    def __init__(self, client_id, dataset, model=None):
        self.client_id = client_id
        self.model = model if model else create_model()
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_confidence = dataset
        self.model_class_confidence = None
        self.model_parameters = None

    def get_parameters(self):
        return self.model_parameters

    def fit(self):
        print(f"Client {self.client_id + 1}: Training...")
        self.model.fit(self.x_train, self.y_train, epochs=50)

        _, client_model_scores = self.model.predict(self.x_confidence, return_class_sums=True)
        confidence_score = np.zeros(client_model_scores.shape[1])
        for j in range(len(self.x_confidence)):
            confidence_score += client_model_scores[j]
        self.model_class_confidence = int(np.argmax(confidence_score))
        self.model_parameters = (self.model_class_confidence, self.model.get_weights(self.model_class_confidence))

    def evaluate(self):
        result = 100 * (self.model.predict(self.x_test) == self.y_test).mean()
        print("Client Result >>>", result)
        return result

    def set_parameters(self, parameters):
        model_class_confidence, aggregated_weights = parameters
        for i in range(len(aggregated_weights)):
            self.model.set_weight(model_class_confidence, i, int(aggregated_weights[i]))
        print(f"Client {self.client_id + 1}: Weights set from server.")
