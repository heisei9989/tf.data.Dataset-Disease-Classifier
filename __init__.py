import tensorflow as tf
import random
import pathlib
import os
import time
import numpy as np
# x_data, y_data
import tensorflow as tf

tf.enable_eager_execution()
print(tf.__version__)

AUTOTUNE = tf.data.experimental.AUTOTUNE #-1

flags = tf.app.flags

flags.DEFINE_string("imagename", "./data/images.tfrec", ".npy filename")
flags.DEFINE_string("labelname", "./data/label.npy", ".npy filename")
flags.DEFINE_string("datadir", "./flower_photos", "path")
flags.DEFINE_string("TB_CP", "./logs/", "tensorboard and checkpoint")
flags.DEFINE_string("model", "./model/", "tensorboard and checkpoint")


flags.DEFINE_integer("steps_per_epoch", 3, "steps")
flags.DEFINE_integer("BATCH_SIZE", 32, "BATCH_SIZE")
flags.DEFINE_integer("epoch", 1, "epoch")

flags.DEFINE_integer("width", 192, "width")
flags.DEFINE_integer("height", 192, "height")

FLAGS = flags.FLAGS

def preprocess_image(image):
	image = tf.image.decode_gif(image) # jpeg and gif color number 1 : gray number 3 :color
	image = tf.image.resize_images(image, [192, 192])
	image = tf.reshape(image, [192,192,3])
	return image

def load_and_preprocess_image(path):
		image = tf.read_file(path)
		return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
	return load_and_preprocess_image(path), label


def change_range(image,label):
	return 2*image-1, label

def main(_):

	if not os.path.exists("./data"):
		os.mkdir("./data")

	if not os.path.exists("./logs"):
		os.mkdir("./logs")

	if not os.path.exists("./model"):
		os.mkdir("./model")
			
	data_root = pathlib.Path(os.path.join(os.getcwd(), FLAGS.datadir))

	### images ###

	all_image_paths = list(data_root.glob("*/*"))
	all_image_paths = [str(path) for path in all_image_paths]

	
	### labels ###

	label_names = sorted(item.name for item in data_root.glob("*/") if item.is_dir())
	label_to_index = dict((name, index) for index, name in enumerate(label_names))
	all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]
	print(len(label_names))
	### tfrecord images (Save) ###

	image_ds = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.read_file)
	tfrec = tf.data.experimental.TFRecordWriter(FLAGS.imagename)
	tfrec.write(image_ds)

	### tfrecord images(Loader) ###

	image_ds = tf.data.TFRecordDataset(FLAGS.imagename).map(preprocess_image)
	
	### np labels (Save) ###

	np.save(FLAGS.labelname, all_image_labels)

	### np labels (Loader) ###

	all_image_labels = np.load(FLAGS.labelname)
	label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

	# Dataset

	image_count = len(all_image_paths)

	ds = tf.data.Dataset.zip((image_ds, label_ds))
	ds = ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
	ds = ds.batch(FLAGS.BATCH_SIZE).prefetch(AUTOTUNE)
	ds = ds.prefetch(buffer_size = AUTOTUNE)
	
	### Direct Method ###

	# ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))
	# image_label_ds = ds.map(load_and_preprocess_from_path_label)

	# ds = image_label_ds.shuffle(buffer_size=image_count)
	# ds = ds.repeat()
	# ds = ds.batch(FLAGS.BATCH_SIZE)
	# ds = ds.prefetch(buffer_size=AUTOTUNE)

	# ds = image_label_ds.apply(
	# tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
	# ds = ds.batch(FLAGS.BATCH_SIZE)
	# ds = ds.prefetch(buffer_size = AUTOTUNE)

	mobile_net = tf.keras.applications.MobileNetV2(input_shape=(FLAGS.width,FLAGS.height, 3), include_top=False)
	mobile_net.trainable = False

	keras_ds = ds.map(change_range)
	image_batch, label_batch = next(iter(keras_ds))
	feature_map_batch = mobile_net(image_batch)

	model = tf.keras.Sequential([
		mobile_net,
		tf.keras.layers.GlobalAveragePooling2D(),
		tf.keras.layers.Dense(len(label_names))])

	logit_batch = model(image_batch).numpy()

	model.compile(optimizer=tf.train.AdamOptimizer(), 
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=["accuracy"])

	model.summary()

	steps_per_epoch=tf.ceil(len(all_image_paths)/FLAGS.BATCH_SIZE).numpy()
	steps_per_epoch


	cb_tb = tf.keras.callbacks.TensorBoard(log_dir = FLAGS.TB_CP, histogram_freq=2)
	cb_es = tf.keras.callbacks.EarlyStopping(patience =5, monitor= "val_loss")

	callbacks = [cb_tb, cb_es]

	model.fit(ds, 
	validation_data = ds,
	epochs=FLAGS.epoch, 
	steps_per_epoch=FLAGS.steps_per_epoch, 
	callbacks=callbacks,
	validation_steps = 1) # tensorflow==1.13.0rc

	model.save(FLAGS.model + "model.h5")
	model.save_weights(FLAGS.model + "weights.h5")

	model.evaluate(ds, batch_size=FLAGS.BATCH_SIZE, steps=1) # need steps


if __name__ == "__main__":
	tf.app.run()