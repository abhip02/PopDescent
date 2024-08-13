def graph_history(history):
	integers = [i for i in range(1, (len(history))+1)]

	ema = []
	avg = history[0]

	ema.append(avg)

	for loss in history:
		avg = (avg * 0.9) + (0.1 * loss)
		ema.append(avg)


	x = [j * rr * (batches * pop_size) for j in integers]
	y = history

	# plot line
	plt.plot(x, ema[:len(history)])
	# plot title/captions
	plt.title("Keras Tuner CIFAR100")
	plt.xlabel("Gradient Steps")
	plt.ylabel("Validation Loss")
	plt.tight_layout()


	print("ema:"), print(ema), print("")
	print("x:"), print(x), print("")
	print("history:"), print(history), print("")


	
	# plt.savefig("TEST_DATA/PD_trial_%s.png" % trial)
	def save_image(filename):
		p = PdfPages(filename)
		fig = plt.figure(1)
		fig.savefig(p, format='pdf') 
		p.close()

	filename = "KerasTuner_CIFAR100_progress_with_reg_line.pdf"
	save_image(filename)

	# plot points too
	plt.scatter(x, history, s=20)

	def save_image(filename):
		p = PdfPages(filename)
		fig = plt.figure(1)
		fig.savefig(p, format='pdf') 
		p.close()

	filename = "KerasTuner_CIFAR100_progress_with_reg__with_points.pdf"
	save_image(filename)


	plt.show(block=True), plt.close()
	plt.close('all')