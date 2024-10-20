function combanz(input_file_path, output_file_path)
    combanzmodule = py.importlib.import_module('combanz');
    combanzmodule.combanz(input_file_path, output_file_path);
end