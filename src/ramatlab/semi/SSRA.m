classdef SSRA
    properties
        pyObj  % Python 对象实例
    end
    
    methods
        % 构造函数
        function obj = SSRA()
            % 创建 Python 类的实例
            obj.pyObj = py.rapython.SSRA();
        end
        
        % 调用 Python 类的 train 方法
        function obj = train(obj, train_file_path, train_rel_path, input_type, alpha, beta, constraints_rate, is_partial_list)
            if nargin < 8
                is_partial_list = true; % 默认值
            end
            if nargin < 7
                constraints_rate = 0.3; % 默认值
            end
            if nargin < 6
                beta = 0.1; % 默认值
            end
            if nargin < 5
                alpha = 0.03; % 默认值
            end
            
            % 调用 Python 的 train 方法
            obj.pyObj.train(train_file_path, train_rel_path, input_type, alpha, beta, constraints_rate, is_partial_list);
        end
        
        % 调用 Python 类的 test 方法
        function test(obj, test_file_path, test_output_path, using_average_w)
            if nargin < 4
                using_average_w = true; % 默认值
            end
            
            % 调用 Python 的 test 方法
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
