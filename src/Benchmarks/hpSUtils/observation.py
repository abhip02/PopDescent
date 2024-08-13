# DATA
import dataset
dataset = dataset.preprocess_dataset()
train_images, train_labels, validation_images, validation_labels, test_images, test_labels = dataset.train_images, dataset.train_labels, dataset.validation_images, dataset.validation_labels, dataset.test_images, dataset.test_labels

lossfn = dataset.lossfn

# observing optimization progress
# unnormalized
def observer(NN_object, tIndices):
	random_batch_test_images, random_batch_test_labels = test_images[tIndices], test_labels[tIndices]	

	test_loss = lossfn(random_batch_test_labels, NN_object(random_batch_test_images))
	ntest_loss = 1/(1+test_loss)

	return test_loss

def graph_history(history, trial, model_string, training_loss_data_string, test_loss_data_string, best_lr_data, best_reg_amount_string):
	integers = [i for i in range(1, (len(history))+1)]
	x = [j * rr for j in integers]
	y = history

	plt.scatter(x, history, s=20)

	plt.title("HPS CIFAR100")
	plt.tight_layout()
	# plt.savefig("TEST_DATA/HP_trial_%s.png" % trial)
	plt.show(block=True), plt.pause(0.5), plt.close()