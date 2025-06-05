"""make_test_suite.py
This file is part of keras2c
Copyright 2020 Rory Conlin
Licensed under LGPLv3
https://github.com/f0uriest/keras2c

Generates automatic test suite for converted code
"""

# Imports
import numpy as np
from keras2c.io_parsing import get_model_io_names
from keras2c.weights2c import Weights2C
import subprocess
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

__author__ = "Rory Conlin"
__copyright__ = "Copyright 2020, Rory Conlin"
__license__ = "MIT"
__maintainer__ = "Rory Conlin, https://github.com/f0uriest/keras2c"
__email__ = "wconlin@princeton.edu"

TEMPLATE_DIR = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), keep_trailing_newline=True)


def make_test_suite(
    model,
    function_name,
    malloc_vars,
    num_tests=10,
    stateful=False,
    verbose=True,
    tol=1e-5,
):
    """Generates code to test the generated C function.

    Generates random inputs to the model and gets the corresponding predictions for them.
    Writes input/output pairs to a C file, along with code to call the generated C function
    and compare the true outputs with the outputs from the generated code.

    Writes the test function to a file `<function_name>_test_suite.c`

    Args:
        model (keras.Model): model being converted to C
        function_name (str): name of the neural net function being generated
        malloc_vars (dict): dictionary of names and values of variables allocated on the heap
        num_tests (int): number of tests to generate
        stateful (bool): whether the model contains layers that maintain state between calls
        verbose (bool): whether to print output
        tol (float): tolerance for passing tests. Tests pass if the maximum error over
            all elements between the true output and generated code output is less than tol

    Returns:
        None
    """

    if verbose:
        print("Writing tests")
    input_shape = []
    model_inputs, model_outputs = get_model_io_names(model)
    num_inputs = len(model_inputs)
    num_outputs = len(model_outputs)

    for i in range(num_inputs):
        temp_input_shape = model.inputs[i].shape
        temp_input_shape = [1 if dim is None else dim for dim in temp_input_shape]
        if stateful:
            temp_input_shape = temp_input_shape[:]
        else:
            temp_input_shape = temp_input_shape[1:]  # Exclude batch dimension
        input_shape.append(temp_input_shape)


    arrays = []
    for i in range(num_tests):
        if i == num_tests // 2 and stateful:
            for layer in model.layers:
                if hasattr(layer, "reset_states") and callable(layer.reset_states):
                    layer.reset_states()
        ct = 0
        while True:
            rand_inputs = []
            for j in range(num_inputs):
                inp_layer = model.inputs[j]
                dt = getattr(inp_layer, "dtype", None)
                if dt is not None and hasattr(dt, "as_numpy_dtype"):
                    dt = dt.as_numpy_dtype
                if dt is not None and np.issubdtype(np.dtype(dt), np.integer):
                    high = 10
                    for layer in model.layers:
                        for node in layer._inbound_nodes:
                            inputs = getattr(node, "input_tensors", [])
                            if not isinstance(inputs, (list, tuple)):
                                inputs = [inputs]
                            if inp_layer in inputs:
                                if hasattr(layer, "input_dim"):
                                    high = layer.input_dim
                                break
                    rand_input = np.random.randint(0, high, size=tuple(input_shape[j]), dtype=np.dtype(dt))
                else:
                    rand_input = 4 * np.random.random(size=tuple(input_shape[j])) - 2
                if not stateful:
                    rand_input = rand_input[np.newaxis, ...]
                rand_inputs.append(rand_input)
            pred_input = rand_inputs if num_inputs > 1 else rand_inputs[0]
            outputs = model.predict(pred_input)
            outputs_concat = np.concatenate([np.ravel(o) for o in outputs]) if isinstance(outputs, list) else outputs
            if np.isfinite(outputs_concat).all():
                break
            ct += 1
            if ct > 20:
                raise Exception("Cannot find inputs to the network that result in a finite output")
        for j in range(num_inputs):
            arrays.append(
                Weights2C.array2c(
                    rand_inputs[j][0, :],
                    f"test{i + 1}_{model_inputs[j]}_input",
                )
            )
        if not isinstance(outputs, list):
            outputs = [outputs]
        for j in range(num_outputs):
            output = outputs[j][0, :]
            arrays.append(
                Weights2C.array2c(
                    output,
                    f"keras_{model_outputs[j]}_test{i + 1}",
                )
            )
            arrays.append(
                Weights2C.array2c(
                    np.zeros(output.shape),
                    f"c_{model_outputs[j]}_test{i + 1}",
                )
            )

    context = {
        "function_name": function_name,
        "arrays": arrays,
        "num_tests": num_tests,
        "num_outputs": num_outputs,
        "model_inputs": model_inputs,
        "model_outputs": model_outputs,
        "malloc_vars": list(malloc_vars),
        "stateful": stateful,
        "tol": tol,
        "half": num_tests // 2,
    }

    with open(function_name + "_test_suite.c", "w") as file:
        file.write(env.get_template("test_suite.c.j2").render(context))
    try:
        subprocess.run(["astyle", "-n", function_name + "_test_suite.c"])
    except FileNotFoundError:
        print(
            f"astyle not found, {function_name}_test_suite.c will not be auto-formatted"
        )
