import subprocess as sp
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def get_gpu_memory_usage():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    memory_free = sum(memory_free_values)

    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = _output_to_list(sp.check_output(command.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_used = sum(memory_used_values)

    return memory_free, memory_used


def get_model_memory_usage(model, unit, batch_size=1):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, layer)
        single_layer_mem = 1
        out_shape = layer.output_shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for shape in out_shape:
            if shape is None:
                continue
            single_layer_mem *= shape
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)

    model_bytes = int(total_memory + internal_model_mem_count) #bytes

    unit_dict = {
        "kibi": 1,
        "mebi": 2,
        "gibi": 3
    }

    if unit in unit_dict:
        model_size = int(model_bytes/(1024 ** unit_dict[unit]))
    else:
        unit = "byte"
        model_size = model_bytes

    return model_size


def get_max_batch_size(model, unit="byte"):
    #Unit: "byte", "kibi", "mebi", "gibi"
    _, gpu_used = get_gpu_memory_usage()
    model_size = get_model_memory_usage(model, unit=unit)
    print(f"GPU memory allocated: {gpu_used}")
    print(f"Model size: {model_size}")

    return gpu_used // model_size


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    model = tf.keras.applications.VGG16(weights=None, input_shape=(32, 32, 3), classes=10)
    print(model.summary())

    model.compile(loss="sparse_categorical_crossentropy")
    max_batch_size = get_max_batch_size(model, unit="mebi")
    print(f"Max batch size: {max_batch_size}")

    model.fit(x_train[:5, ...], y_train[:5], verbose=True)
