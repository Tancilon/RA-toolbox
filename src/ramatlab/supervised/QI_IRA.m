classdef QI_IRA
    methods(Static)
        function qi_ira(input_file_path, output_file_path, input_rel_path, input_type)
            % MATLAB 封装 Python 的 qi_ira 函数
            
            if nargin < 4
                input_type = py.rapython.InputType.SCORE;  % 使用默认 InputType.SCORE
            end
            
            % 调用 Python 的 qi_ira 函数
            py.rapython.ira.qi_ira(input_file_path, output_file_path, input_rel_path, input_type);
        end
    end
end
