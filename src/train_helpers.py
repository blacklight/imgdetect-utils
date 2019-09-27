import json
import os

import numpy as np
import matplotlib.pyplot as plt


def _load_dataset_file(dataset_file, split_percentage=0.7):
    dataset = np.load(dataset_file).get('data')
    labels = np.load(dataset_file).get('labels')
    classes = np.load(dataset_file).get('classes')

    items = np.asarray([
        [data, labels[i]]
        for i, data in enumerate(dataset)
    ])

    np.random.shuffle(items)

    # dataset = np.asarray([item['data'] for item in items])
    # labels = np.asarray([item['label'] for item in items], dtype=np.uint8)
    pivot = int((dataset.shape[0] - 1) * split_percentage)

    # return (dataset[:pivot + 1], labels[:pivot + 1]), (dataset[pivot + 1:], labels[pivot + 1:]), classes
    return items[:pivot + 1], items[pivot + 1:], classes


def load_data(*dataset_files, split_percentage=0.7):
    train_set = []
    test_set = []
    classes = None

    for dataset_file in dataset_files:
        _train_set, _test_set, _classes = _load_dataset_file(dataset_file, split_percentage=split_percentage)
        train_set.extend(_train_set)
        test_set.extend(_test_set)

        if classes is None:
            classes = _classes

    return np.asarray(train_set), np.asarray(test_set), classes


def plot_image_and_predictions(i, predictions, classes, true_label, img):
    predictions, true_label, img = predictions[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    # noinspection PyUnresolvedReferences
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = int(np.argmax(predictions))
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(classes[predicted_label],
                                         100 * np.max(predictions),
                                         classes[int(true_label)]), color=color)


def plot_value_array(i, predictions, true_label):
    predictions, true_label = predictions[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(predictions)), predictions, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
def plot_results(images, labels, predictions, classes, rows, cols):
    n_images = rows * cols
    plt.figure(figsize=(2 * 2 * cols, 2 * rows))

    for i in range(n_images):
        plt.subplot(rows, 2 * cols, 2 * i + 1)
        plot_image_and_predictions(i, predictions, classes, labels, images)
        plt.subplot(rows, 2 * cols, 2 * i + 2)
        plot_value_array(i, predictions, labels)

    plt.show()


def get_predicted_label(image, classes, model):
    predictions = model.predict(np.asarray([image]))
    predicted_label = np.argmax(predictions)
    return classes[predicted_label]


def save_model(model, model_file, weights_file):
    with open(model_file, 'w') as f:
        f.write(model.to_json())
    model.save_weights(weights_file)


def export_model(model, model_file, properties_file=None, classes=None, input_size=None, color_convert=None):
    import tensorflow as tf
    # noinspection PyUnresolvedReferences,PyPep8Naming
    from tensorflow.keras import backend as K

    # noinspection PyUnresolvedReferences
    frozen_graph = freeze_session(K.get_session(),
                                  output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, os.path.dirname(model_file),
                         os.path.basename(model_file), as_text=False)

    if properties_file:
        properties = {}

        if classes is not None:
            if isinstance(classes, np.ndarray):
                classes = classes.tolist()
            properties['classes'] = classes

        if input_size is not None:
            properties['input_size'] = [*input_size]
        if color_convert is not None:
            properties['color_convert'] = color_convert

        if properties:
            with open(properties_file, 'w') as f:
                json.dump(properties, f)


def load_model(model_file, weights_file):
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.models import model_from_json

    with open(model_file, 'r') as f:
        model = model_from_json(f.read())
        model.load_weights(weights_file)
        return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    import tensorflow as tf
    from tensorflow.python.framework.graph_util import convert_variables_to_constants

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def predict(model, img, properties_file=None, resize=None, classes=None, color_convert=None):
    import cv2

    if isinstance(img, str):
        img = cv2.imread(img)

    if properties_file:
        with open(properties_file) as f:
            properties = json.load(f)

        resize = properties.get('input_size', resize)
        classes = properties.get('classes', classes)
        color_convert = properties.get('color_convert', color_convert)

    if color_convert:
        if isinstance(color_convert, str):
            color_convert = getattr(cv2, color_convert)

        img = cv2.cvtColor(img, color_convert)

    if resize:
        img = cv2.dnn.blobFromImage(img, size=tuple(resize), mean=0.5)
    else:
        img = cv2.dnn.blobFromImage(img, mean=0.5)

    model.setInput(img)
    output = model.forward()
    prediction = int(np.argmax(output))

    if classes:
        prediction = classes[prediction]

    return prediction
