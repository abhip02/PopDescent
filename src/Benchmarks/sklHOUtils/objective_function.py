import numpy as np

# DATA
import dataset
dataset = dataset.preprocess_dataset()

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels, dataset.test_images, dataset.test_labels


def return_objective(sklHO, training_parameters):
    def objective_func(params):
        model = sklHO(training_parameters, params)

        batch_size = training_parameters.batch_size
        batches = training_parameters.batches

        # Train and evaluate the model (modify this according to your dataset and training process)
        train_epochs = training_parameters.epochs
        indices = np.random.choice(len(train_images), size = (batch_size*batches, ), replace=False)
        vIndices = np.random.choice(len(validation_images), size = (batch_size*10, ), replace=False)

        # FM dataset
        random_batch_train_images, random_batch_train_labels = train_images[indices], train_labels[indices]
        random_batch_validation_images, random_batch_validation_labels = validation_images[vIndices], validation_labels[vIndices]

        history = model.fit(random_batch_train_images, random_batch_train_labels, batch_size=batch_size, validation_data=(random_batch_validation_images, random_batch_validation_labels), epochs=train_epochs)

        # Access the validation accuracy
        validation_accuracy = history.history['val_accuracy'][-1]

        # Hyperopt minimizes the objective function, so negate the metric you want to maximize
        return -validation_accuracy
    
    return objective_func