function rrf(input_file_path, output_file_path)
    rrfmodule = py.importlib.import_module('rrf');
    rrfmodule.rrf(input_file_path, output_file_path);
end