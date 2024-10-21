classdef AggRankDE
    properties
        pyObj  % Python 对象实例
    end
    
    methods
        % 构造函数
        function obj = AggRankDE(np_val, max_iteration, cr, f, input_type, n)
            if nargin < 6
                n = py.None;
            end
            if nargin < 5
                input_type = py.None;
            end
            if nargin < 4
                f = 0.5;
            end
            if nargin < 3
                cr = 0.9;
            end
            if nargin < 2
                max_iteration = 100;
            end
            if nargin < 1
                np_val = 50;
            end
            
            % 创建 Python 类的实例
            obj.pyObj = py.rapython.aggrankde.AggRankDE(np_val, max_iteration, cr, f, input_type, n);
        end
        
        % 调用 Python 类的 train 方法
        function train(obj, train_file_path, train_rel_path, input_type)
            if nargin < 4
                input_type = py.None;
            end
            obj.pyObj.train(train_file_path, train_rel_path, input_type);
        end
        
        % 调用 Python 类的 test 方法
        function test(obj, test_file_path, test_output_path, using_average_w)
            if nargin < 4
                using_average_w = true;
            end
            obj.pyObj.test(test_file_path, test_output_path, using_average_w);
        end
    end
end
