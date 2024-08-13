def train_KT_model(model, training_parameters, train_images, train_labels, validation_images, validation_labels, callback):
 	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = training_parameters.train_epochs
	model.fit(train_images, train_labels, batch_size=training_parameters.batch_size, validation_data=(validation_images, validation_labels), epochs=training_parameters.train_epochs, callbacks=[callback])
	
	return
