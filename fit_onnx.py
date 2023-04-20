import onnx
from onnxsim import simplify


def change_input_output_names(model, input_name, output_name):
    # works only for nets with one input and output each

    old_input_names = []
    old_output_names = []
    # Update the input and output names
    for input in model.graph.input:
        old_input_names.append(input.name)

    for output in model.graph.output:
        old_output_names.append(output.name)

    # change input and output names
    for i in range(len(model.graph.node)):
        for j in range(len(model.graph.node[i].input)):
            if model.graph.node[i].input[j] in old_input_names:
                model.graph.node[i].input[j] = input_name

        for j in range(len(model.graph.node[i].output)):
            if model.graph.node[i].output[j] in old_output_names:
                model.graph.node[i].output[j] = output_name

    for i in range(len(model.graph.input)):
        if model.graph.input[i].name in old_input_names:
            model.graph.input[i].name = input_name

    for i in range(len(model.graph.output)):
        if model.graph.output[i].name in old_output_names:
            model.graph.output[i].name = output_name

    print("Changed the input and output names successfully.")
    return model


# Specify the path to your ONNX model file
onnx_model_path = 'tcn_model.onnx'

# Load the ONNX model
model = onnx.load(onnx_model_path)

# Call the function to change the input and output name from the model
input_name = 'input'
output_name = 'output'
model = change_input_output_names(model, input_name, output_name)

# Save the modified ONNX model
onnx.checker.check_model(model)
onnx.save(model, 'tcn_model_mod.onnx')

# simplify the model structure 
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"

# Save the simplified ONNX model
onnx.checker.check_model(model_simp)
onnx.save(model_simp, 'tcn_model_mod_sim.onnx')