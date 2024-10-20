function postndcg(input_file_path, output_file_path, input_type)
    postndcgmodule = py.importlib.import_module('postndcg');
    postndcgmodule.postndcg(input_file_path, output_file_path, input_type);
end