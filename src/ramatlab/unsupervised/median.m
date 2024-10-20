function median(input_file_path, output_file_path)
    medianmodule = py.importlib.import_module('median');
    medianmodule.median(input_file_path, output_file_path);
end