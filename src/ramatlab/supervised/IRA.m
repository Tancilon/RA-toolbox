classdef IRA
    methods(Static)
        function ira(input_file_path, output_file_path, input_rel_path, mode, input_type)
            % MATLAB 封装 Python 的 ira 函数
            
            if nargin < 5
                input_type = py.rapython.InputType.SCORE;  % 使用默认 InputType.SCORE
            end
            if nargin < 4
                mode = py.rapython.MethodType.RANK;  % 使用默认 MethodType.RANK
            end
            
            % 调用 Python 的 ira 函数
            py.ira(input_file_path, output_file_path, input_rel_path, mode, input_type);
        end
    end
end
