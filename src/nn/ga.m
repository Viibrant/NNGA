% Put in seperate file

function y = Shubert1(x)
    n = length(x);
    sum1 = 0;
    for i = 1:n
      for j = 1:5
        sum1 = sum1 + j * cos((j + 1) * x(i) + j);
      end
    end
    y = sum1;
  end

% Put in seperate file

options = optimoptions('ga', 'PopulationSize', 100, 'MutationFcn',
  'mutationadaptfeasible','CrossoverFcn', 'crossoverscattered',
  'EliteCount', 2, 'SelectionFcn', 'selectionstochunif', 'PlotFcn',
  @gaplotbestf);

nvars = 2; % Number of variables
lb = [-10, -10]; % Lower bounds
ub = [10, 10]; % Upper bounds
[x,fval] = ga(@Shubert1, nvars, [], [], [], [], lb, ub, [], options);

options = optimoptions('ga', 'PlotFcn', @gaplotbestf);