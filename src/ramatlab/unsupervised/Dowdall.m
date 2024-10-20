function dowdall(input_file_path, output_file_path)
    dowdallmodule = py.importlib.import_module('dowdall');
    dowdallmodule.dowdall(input_file_path, output_file_path);
end