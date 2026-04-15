from src import utils

def evaluate_model(model, dataset, batch_size):
	print(""), print(""), print("Evaluating models on test data after randomization")
	random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = utils.random_eval_set_generator(dataset, batch_size)
	train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
	test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]
	
	return train_loss, test_loss