function cg(input_file_path, output_file_path)
    cgmodule = py.importlib.import_module('cg');
    cgmodule.cg(input_file_path, output_file_path);
end