% Initialize the fixed DCM Buck model and period-normalized output grid.

script_dir = fileparts(mfilename("fullpath"));
model_name = "buck_ideal_hybrid";
model_path = fullfile(script_dir, model_name + ".slx");
schema_version = "hybrid_event_matlab_v3";

if ~isfile(model_path)
    error("BuckNeuralODE:MissingModel", ...
        "找不到 Simulink 模型：%s", model_path);
end

% Fixed circuit parameters.
L = 500e-6;
C = 100e-6;
R = 80;
fs = 20e3;
D = 0.5;

% Period-normalized output grid.
samples_per_cycle = 100;
Ts = 1 / fs;
dt = Ts / samples_per_cycle;
