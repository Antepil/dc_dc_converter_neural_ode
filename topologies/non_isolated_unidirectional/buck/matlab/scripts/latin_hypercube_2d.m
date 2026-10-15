function factors = latin_hypercube_2d(count, current_range, voltage_range)
%LATIN_HYPERCUBE_2D Generate a two-dimensional Latin hypercube sample.
%   This implementation uses only base MATLAB and consumes the active RNG
%   stream. The caller is responsible for setting the split-level seed.

    arguments
        count (1, 1) double {mustBeInteger, mustBePositive}
        current_range (1, 2) double {mustBeFinite}
        voltage_range (1, 2) double {mustBeFinite}
    end

    if current_range(2) <= current_range(1) || ...
            voltage_range(2) <= voltage_range(1)
        error("BuckNeuralODE:InvalidLHSRange", ...
            "LHS ranges must be strictly increasing.");
    end

    currentBins = ((0:count - 1)' + rand(count, 1)) / count;
    voltageBins = ((0:count - 1)' + rand(count, 1)) / count;
    currentBins = currentBins(randperm(count));
    voltageBins = voltageBins(randperm(count));

    factors = [ ...
        current_range(1) + currentBins * diff(current_range), ...
        voltage_range(1) + voltageBins * diff(voltage_range)];
end
