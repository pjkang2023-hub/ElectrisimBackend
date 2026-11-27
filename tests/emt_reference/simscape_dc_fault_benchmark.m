function simscape_dc_fault_benchmark()
% SIMSCAPE_DC_FAULT_BENCHMARK  Reference waveforms for the DC fault study.
%
% Builds a Simscape (Foundation Library) model of the circuit described in
% dc_fault_benchmark_params.json - the same file tests/test_dc_fault.py builds
% Electrisim's circuit from - simulates the fault, and writes the branch
% currents to dc_fault_benchmark.csv, which the Python test compares against.
% Run it again when the solver or the benchmark changes; needs Simscape.
%
% Every capacitor starts at v0 and every inductor at 0 A; the fault is
% present from t = 0. Currents (A), each in the direction named:
%   i_fault   DC bus B to the negative pole, through the fault
%   i_cap     DC-link capacitor into DC bus A
%   i_bridge  the diode bridge's DC+ into DC bus A
%   i_cable   DC bus A to DC bus B, through the cable
%   i_batt    the battery into DC bus B
%
% Simscape's diode needs a forward voltage above zero: 1 uV, where Electrisim's has none.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'dc_fault_benchmark_params.json')));
fl = 'fl_lib/Electrical/Electrical Elements/';
src = 'fl_lib/Electrical/Electrical Sources/';

mdl = 'dc_fault_benchmark';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);
cleanup = onCleanup(@() close_system(mdl, 0));
count = 0;
nodes = {};   % each node: the first port wired to it, which later ports are wired to

ref = place([fl 'Electrical Reference']);
gnd = node();
connect(gnd, ports(ref).LConn(1));
solver = place('nesl_utility/Solver Configuration');
connect(gnd, ports(solver).RConn(1));

% The grid behind its impedance, with its neutral held at the DC link's midpoint.
w = 2 * pi * p.f_hz;
bridge_p = node();
neutral = node();
two(res(p.r_bias), neutral, bridge_p);
two(res(p.r_bias), neutral, gnd);
for k = 0:2
    e = place([src 'AC Voltage Source']);
    % Electrisim's e = E cos(w t + phi) is Simscape's E sin(w t + phi + 90 deg).
    set_param(e, 'amp', num2str(sqrt(2 / 3) * p.v_ll, 17), 'frequency', num2str(p.f_hz, 17), ...
        'shift', num2str(p.phase_deg + 90 - 120 * k, 17));
    a = node();
    b = node();
    two(e, a, neutral);                       % + at a, - at the neutral
    two(res(p.r_ac), a, b);
    x = node();
    two(ind(p.x_ac / w), b, x);
    two(diode(), x, bridge_p);                % upper: anode the phase, cathode DC+
    two(diode(), gnd, x);                     % lower: anode DC-, cathode the phase
end

% DC bus A: the bridge, the DC-link capacitor, the cable's near end.
bus_a = node();
r_bridge = res(p.r_dc);
two(r_bridge, bridge_p, bus_a);
c_inner = node();
c_mid = node();
two(cap(p.c_link, p.v0), c_inner, gnd);
r_cap = res(p.esr);
two(r_cap, c_inner, c_mid);
two(ind(p.esl), c_mid, bus_a);

% The cable, as a pi section.
bus_b = node();
two(cap(p.cable_c / 2, p.v0), bus_a, gnd);
two(cap(p.cable_c / 2, p.v0), bus_b, gnd);
cable_mid = node();
r_cable = res(p.cable_r);
two(r_cable, bus_a, cable_mid);
two(ind(p.cable_l), cable_mid, bus_b);

% DC bus B: the battery behind its R and L, and the fault.
batt = place([src 'DC Voltage Source']);
set_param(batt, 'v0', num2str(p.batt_e, 17));
batt_p = node();
two(batt, batt_p, gnd);
batt_mid = node();
r_batt = res(p.batt_r);
two(r_batt, batt_p, batt_mid);
two(ind(p.batt_l), batt_mid, bus_b);
r_fault = res(p.r_fault);
two(r_fault, bus_b, gnd);

set_param(mdl, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
    'StopTime', num2str(p.t_end, 17), 'Solver', 'ode23t', 'MaxStep', '1e-6', ...
    'RelTol', '1e-7', 'AbsTol', '1e-6');
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on');
log = out.get('simlog');

% Resampled onto a fixed grid: fine through the capacitor's discharge.
t = unique([0:5e-6:0.01, 0.01:5e-5:p.t_end])';
current = @(h) interp1(log.(get_param(h, 'Name')).i.series.time, ...
    log.(get_param(h, 'Name')).i.series.values('A'), t);
tbl = table(t, current(r_fault), current(r_cap), current(r_bridge), current(r_cable), current(r_batt), ...
    'VariableNames', {'t', 'i_fault', 'i_cap', 'i_bridge', 'i_cable', 'i_batt'});
writetable(tbl, fullfile(here, 'dc_fault_benchmark.csv'));
fprintf('Wrote %d rows; fault current peak %.4f kA at %.4f ms\n', height(tbl), ...
    max(tbl.i_fault) / 1e3, 1e3 * t(find(tbl.i_fault == max(tbl.i_fault), 1)));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', mdl, count));
    end
    function pp = ports(h)
        pp = get_param(h, 'PortHandles');
    end
    function wire(a, b)
        add_line(mdl, a, b, 'autorouting', 'off');
    end
    function n = node()
        nodes{end + 1} = [];
        n = numel(nodes);
    end
    function two(h, a, b)
        % Block h's + port on node a, its - port on node b.
        pp = ports(h);
        connect(a, pp.LConn(1));
        connect(b, pp.RConn(1));
    end
    function connect(n, port)
        if isempty(nodes{n})
            nodes{n} = port;
        else
            wire(nodes{n}, port);
        end
    end
    function h = res(r)
        h = place([fl 'Resistor']);
        set_param(h, 'R', num2str(r, 17));
    end
    function h = ind(l)
        h = place([fl 'Inductor']);
        set_param(h, 'l', num2str(l, 17), 'r', '0', 'g', '0', ...
            'i_L_specify', 'on', 'i_L_priority', 'High', 'i_L', '0');
    end
    function h = cap(c, v0)
        h = place([fl 'Capacitor']);
        set_param(h, 'c', num2str(c, 17), 'r', '0', 'g', '0', ...
            'vc_specify', 'on', 'vc_priority', 'High', 'vc', num2str(v0, 17));
    end
    function h = diode()
        h = place([fl 'Diode']);
        set_param(h, 'Vf', '1e-6', 'Ron', num2str(p.r_on, 17), 'Goff', num2str(1 / p.r_off, 17));
    end
end
