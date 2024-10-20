function combmin(input_file_path, output_file_path)
    combminmodule = py.importlib.import_module('combmin');
    combminmodule.combmin(input_file_path, output_file_path);
end