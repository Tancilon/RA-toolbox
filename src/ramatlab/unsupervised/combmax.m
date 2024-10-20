function combmax(input_file_path, output_file_path)
    combmaxmodule = py.importlib.import_module('combmax');
    combmaxmodule.combmax(input_file_path, output_file_path);
end