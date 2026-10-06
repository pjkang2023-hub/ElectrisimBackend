function simscape_emt_vsc_benchmark()
% SIMSCAPE_EMT_VSC_BENCHMARK  Reference waveforms for the VSC's average-value model.
%
% Builds the circuit in emt_vsc_benchmark_params.json in Simscape (Foundation
% Library): a 400 V grid behind R_g and L_g (R_p across L_g), filter
% capacitors at the converter's AC bus, its averaged phase voltages behind its
% reactor, and on its DC side its DC-link capacitor, the current its AC side
% delivers (-p / v_dc) and a resistive load, stepped up 50 % at t_step. Its
% controller - PLL, dq current control, DC voltage loop, Q at zero, current
% limit - runs in a MATLAB Function block every dt, as Electrisim's: it
% measures at each step and holds its outputs over the next. Both start in
% the circuit's steady state, which this script computes as
% tests/test_emt_converters.py does. Writes emt_vsc_load_step.csv: the DC
% link's voltage and the power the converter delivers to its AC bus.
% Needs Simscape.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_vsc_benchmark_params.json')));
s = steady(p);
w = 2 * pi * p.f_hz;
q = p.vsc;
mdl = 'emt_vsc_load_step';
fl = 'fl_lib/Electrical/Electrical Elements/';
srcs = 'fl_lib/Electrical/Electrical Sources/';
sens = 'fl_lib/Electrical/Electrical Sensors/';
if bdIsLoaded(mdl), close_system(mdl, 0); end
% The MATLAB Function block's build files in a temporary folder, not here.
gen = Simulink.fileGenControl('getConfig');
build = fullfile(tempdir, mdl);
Simulink.fileGenControl('set', 'CacheFolder', build, 'CodeGenFolder', build, 'createDir', true);
restore = onCleanup(@() Simulink.fileGenControl('setConfig', 'config', gen));
new_system(mdl);
cleanup = onCleanup(@() close_system(mdl, 0));
count = 0;
nodes = {};
P = @(h) get_param(h, 'PortHandles');

gnd = node();
connect(gnd, P(place([fl 'Electrical Reference'])).LConn(1));
connect(gnd, P(place('nesl_utility/Solver Configuration')).RConn(1));

% The controller: its inputs sampled every dt, its outputs held over the next.
ctrl = place('simulink/User-Defined Functions/MATLAB Function');
set_param(ctrl, 'Name', 'ctrl');
ctrl = [mdl '/ctrl'];
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', ctrl);
chart.Script = controller_code(p, s);
mux = place('simulink/Signal Routing/Mux');
set_param(mux, 'Inputs', '8');
zoh = place('simulink/Discrete/Zero-Order Hold');
set_param(zoh, 'SampleTime', num2str(p.dt, 17));
% Its measurements 1 ns late: Simscape takes its network's outputs to depend
% on its inputs at once - an algebraic loop through the controller - though
% each signal measured is a state, the same 1 ns before.
lag = place('simulink/Continuous/Transport Delay');
set_param(lag, 'DelayTime', '1e-9', 'InitialOutput', '0');
add_line(mdl, P(mux).Outport(1), P(lag).Inport(1));
add_line(mdl, P(lag).Outport(1), P(zoh).Inport(1));
add_line(mdl, P(zoh).Outport(1), P(ctrl).Inport(1));
dmx = place('simulink/Signal Routing/Demux');
set_param(dmx, 'Outputs', '4');
add_line(mdl, P(ctrl).Outport(1), P(dmx).Inport(1));

ang = -(0:2) * 2 * pi / 3;
g = p.grid;
c_f = zeros(1, 3); l_c = zeros(1, 3);
for k = 1:3
    % The grid: cos(w t + angle) behind R_g, and L_g with R_p across it.
    sine = place('simulink/Sources/Sine Wave');
    set_param(sine, 'Amplitude', num2str(s.E, 17), 'Frequency', num2str(w, 17), 'Phase', num2str(ang(k) + pi / 2, 17));
    a = node(); b = node(); x = node(); pcc = node();
    controlled(place([srcs 'Controlled Voltage Source']), P(sine).Outport(1), 'V', gnd, a);
    two(res(s.r_g), a, b);
    two(ind(s.l_g, real(s.Ig * exp(1i * ang(k)))), b, pcc);
    two(res(g.r_p), b, pcc);
    c_f(k) = cap(p.c_f, real(s.V * exp(1i * ang(k))));
    two(c_f(k), pcc, gnd);
    vs = sensor([sens 'Voltage Sensor'], 'V', mux, k);
    two_sensor(vs, pcc, gnd);
    % The converter's phase: its averaged voltage, its reactor, its current into its AC bus.
    y = node();
    controlled(place([srcs 'Controlled Voltage Source']), P(dmx).Outport(k), 'V', gnd, x);
    two(res(q.r), x, y);
    cs = sensor([sens 'Current Sensor'], 'A', mux, 3 + k);
    l_c(k) = ind(q.l, real(s.Ic * exp(1i * ang(k))));
    z = node();
    two(l_c(k), y, z);
    two_sensor(cs, z, pcc);
end

% Its DC side: the link capacitor, the current its AC side delivers, its load.
dcp = node(); ldn = node(); mid = node(); stp = node();
c_link = cap(q.c_link, q.v_dc);
two(c_link, dcp, gnd);
controlled(place([srcs 'Controlled Current Source']), P(dmx).Outport(4), 'A', gnd, dcp);
vs = sensor([sens 'Voltage Sensor'], 'V', mux, 7);
two_sensor(vs, dcp, gnd);
cs = sensor([sens 'Current Sensor'], 'A', mux, 8);
two_sensor(cs, dcp, mid);
two(res(q.r_dc), mid, ldn);
lx = node();
% (The DC inductor after its resistor: the sensor reads the same current.)
two(ind(q.l_dc, s.i_dc), ldn, lx);
two(res(p.load.r_load), lx, gnd);
sw = place([fl 'Switch']);
set_param(sw, 'R_closed', num2str(p.switch_r_on, 17), 'G_open', num2str(1 / p.switch_r_off, 17), 'Threshold', '0.5');
step = place('simulink/Sources/Step');
set_param(step, 'Time', num2str(p.load.t_step, 17), 'Before', '0', 'After', '1');
cv = place('nesl_utility/Simulink-PS Converter');
add_line(mdl, P(step).Outport(1), P(cv).Inport(1));
add_line(mdl, P(cv).RConn(1), P(sw).RConn(1));
pp = P(sw);
connect(lx, pp.LConn(1));
connect(stp, pp.RConn(2));
two(res(p.load.r_step), stp, gnd);

set_param(mdl, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
    'StopTime', num2str(p.t_end, 17), 'Solver', 'ode23t', 'MaxStep', num2str(p.dt, 17), 'RelTol', '1e-7', 'AbsTol', '1e-6');
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on', 'TimeOut', 600);
log = out.get('simlog');
t = (0:p.dt:p.t_end)';
v_dc = series(log, c_link, 'v', t);
pw = zeros(size(t));
names = {'t', 'v_dc', 'p_ac'};
% The power into its AC bus: each phase's PCC voltage times its current.
for k = 1:3
    pw = pw + series(log, c_f(k), 'v', t) .* series(log, l_c(k), 'i', t);
end
tbl = array2table([t, v_dc, pw], 'VariableNames', names);
writetable(tbl, fullfile(here, 'emt_vsc_load_step.csv'));
fprintf('Wrote emt_vsc_load_step.csv: %d rows\n', height(tbl));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', mdl, count));
    end
    function k = node()
        nodes{end + 1} = [];
        k = numel(nodes);
    end
    function connect(k, port)
        if isempty(nodes{k})
            nodes{k} = port;
        else
            add_line(mdl, nodes{k}, port, 'autorouting', 'off');
        end
    end
    function two(h, a, b)
        pp_ = P(h);
        connect(a, pp_.LConn(1));
        connect(b, pp_.RConn(1));
    end
    function two_sensor(h, a, b)
        % A sensor's + on a, its - on b; its measurement is RConn(1).
        pp_ = P(h);
        connect(a, pp_.LConn(1));
        connect(b, pp_.RConn(2));
    end
    function h = sensor(lib, unit, mx, port)
        h = place(lib);
        cv_ = place('nesl_utility/PS-Simulink Converter');
        set_param(cv_, 'Unit', unit);
        add_line(mdl, P(h).RConn(1), P(cv_).LConn(1));
        add_line(mdl, P(cv_).Outport(1), P(mx).Inport(port));
    end
    function controlled(h, from, unit, a, b)
        % A controlled source driven by a Simulink signal: its + (or its
        % head, for a current source) on b, its - (tail) on a.
        cv_ = place('nesl_utility/Simulink-PS Converter');
        set_param(cv_, 'Unit', unit);
        add_line(mdl, from, P(cv_).Inport(1));
        pp_ = P(h);
        add_line(mdl, P(cv_).RConn(1), pp_.RConn(1));
        connect(b, pp_.LConn(1));
        connect(a, pp_.RConn(2));
    end
    function h = res(r)
        h = place([fl 'Resistor']);
        set_param(h, 'R', num2str(r, 17));
    end
    function h = ind(l, i0)
        h = place([fl 'Inductor']);
        set_param(h, 'l', num2str(l, 17), 'r', '0', 'g', '0', ...
            'i_L_specify', 'on', 'i_L_priority', 'High', 'i_L', num2str(i0, 17));
    end
    function h = cap(c, v0)
        h = place([fl 'Capacitor']);
        set_param(h, 'c', num2str(c, 17), 'r', '0', 'g', '0', ...
            'vc_specify', 'on', 'vc_priority', 'High', 'vc', num2str(v0, 17));
    end
end

function y = series(log, h, var, t)
s = log.(get_param(h, 'Name')).(var).series;
[tt, k] = unique(s.time, 'last');
vals = s.values;
y = interp1(tt, vals(k), t);
end

function s = steady(p)
% Its steady state: the DC load's current, then its AC bus voltage V and
% current Ic (in phase with V: Q = 0) by fixed point, as the test does.
g = p.grid; q = p.vsc;
w = 2 * pi * p.f_hz;
z = g.v_ll ^ 2 / g.s_sc;
s.r_g = z / sqrt(1 + g.xr ^ 2);
x_g = g.xr * s.r_g;
s.l_g = x_g / w;
zg = s.r_g + 1i * x_g * g.r_p / (g.r_p + 1i * x_g);
s.E = sqrt(2 / 3) * g.v_ll;
s.i_dc = q.v_dc / (q.r_dc + p.load.r_load);
p_dc = q.v_dc * s.i_dc;
V = s.E;
for it = 1:200
    a = abs(V);
    x = (-a + sqrt(a * a - 4 * q.r * p_dc / 1.5)) / (2 * q.r);
    Ic = x * V / a;
    V = s.E - zg * (1i * w * p.c_f * V - Ic);
end
s.V = V; s.Ic = Ic;
s.Ig = (s.E - V) / zg;
s.Ec = V + (q.r + 1i * w * q.l) * Ic;
end

function code = controller_code(p, s)
% The MATLAB Function block's code: Electrisim's VscAverage.control, with
% its constants and starting state written in.
q = p.vsc;
w0 = 2 * pi * p.f_hz;
v_ll = p.grid.v_ll;
c = struct();
c.w0 = w0; c.dt = p.dt; c.r = q.r; c.l = q.l;
c.i_max = q.i_limit_pu * sqrt(2) * q.s_rated / (sqrt(3) * v_ll);
c.v_nom_peak = sqrt(2 / 3) * v_ll;
% Its start: as VscAverage._start.
th = angle(s.V);
v_pk = abs(s.V);
c.theta0 = th;
c.kp_pll = 2 * 0.7 * (2 * pi * 20) / v_pk; c.ki_pll = (2 * pi * 20) ^ 2 / v_pk;
a_c = 2 * pi * 500;
c.kp_i = a_c * q.l; c.ki_i = c.kp_i * a_c / 10;
i_d = abs(s.Ic) * cos(angle(s.Ic) - th); i_q = abs(s.Ic) * sin(angle(s.Ic) - th);
e_d = abs(s.Ec) * cos(angle(s.Ec) - th); e_q = abs(s.Ec) * sin(angle(s.Ec) - th);
c.int_d0 = e_d - (v_pk + q.r * i_d - w0 * q.l * i_q);
c.int_q0 = e_q - (q.r * i_q + w0 * q.l * i_d);
w_v = 2 * pi * 30;
c.kp_v = 2 * 0.7 * w_v * q.c_link; c.ki_v = w_v * w_v * q.c_link;
c.v_dc_ref = q.v_dc;
c.q_out_ref = -1.5 * v_pk * i_q;
c.int_v0 = -1.5 * v_pk * i_d / q.v_dc - s.i_dc;
e0 = real(s.Ec * exp(-1i * (0:2) * 2 * pi / 3));
c.e0 = sprintf('[%.17g; %.17g; %.17g]', e0);
c.i_dc0 = s.i_dc;
lines = {
'function y = ctrl(u)'
'% u: PCC voltages (3), converter currents into the PCC (3), v_dc, DC load current.'
'% y: the converter''s phase voltages (3), the DC current into its link.'
'persistent th w int_pll int_d int_q int_v started'
'if isempty(started)'
'    th = C_theta0; w = C_w0; int_pll = 0; int_d = C_int_d0; int_q = C_int_q0; int_v = C_int_v0; started = false;'
'end'
'if ~started'
'    % Its first sample, at t = 0: the steady state''s voltages.'
'    started = true;'
'    y = [C_e0; C_i_dc0];'
'    return'
'end'
'dt = C_dt;'
'v = u(1:3); i = u(4:6); v_dc = u(7); i_load = u(8);'
'theta = th + w * dt;'
'a = theta - [0; 2; 4] * pi / 3;'
'v_d = 2 / 3 * sum(v .* cos(a)); v_q = -2 / 3 * sum(v .* sin(a));'
'i_d = 2 / 3 * sum(i .* cos(a)); i_q = -2 / 3 * sum(i .* sin(a));'
'int_pll = int_pll + C_ki_pll * v_q * dt;'
'w = C_w0 + C_kp_pll * v_q + int_pll;'
'vd_ = max(v_d, 0.05 * C_v_nom_peak);'
'err = C_v_dc_ref - v_dc;'
'p_in = v_dc * (i_load + C_kp_v * err + int_v);'
'id_ref = -p_in / (1.5 * vd_);'
'iq_ref = -C_q_out_ref / (1.5 * vd_);'
'limited = false;'
'if abs(id_ref) > C_i_max, id_ref = sign(id_ref) * C_i_max; limited = true; end'
'iq_max = sqrt(max(C_i_max ^ 2 - id_ref ^ 2, 0));'
'if abs(iq_ref) > iq_max, iq_ref = sign(iq_ref) * iq_max; limited = true; end'
'if ~limited, int_v = int_v + C_ki_v * err * dt; end'
'ed = v_d + C_r * i_d - w * C_l * i_q + C_kp_i * (id_ref - i_d) + int_d;'
'eq = v_q + C_r * i_q + w * C_l * i_d + C_kp_i * (iq_ref - i_q) + int_q;'
'e_max = max(v_dc, 0) / sqrt(3);'
'mag = hypot(ed, eq);'
'if mag > e_max'
'    ed = ed * e_max / mag; eq = eq * e_max / mag;'
'else'
'    int_d = int_d + C_ki_i * (id_ref - i_d) * dt;'
'    int_q = int_q + C_ki_i * (iq_ref - i_q) * dt;'
'end'
'b = theta + 0.5 * w * dt - [0; 2; 4] * pi / 3;'
'e = ed * cos(b) - eq * sin(b);'
'th = theta;'
'y = [e; -sum(e .* i) / max(v_dc, 1)];'
};
code = strjoin(lines, newline);
% Longest names first: none is then a prefix of one still to come.
names = fieldnames(c);
[~, order] = sort(-cellfun(@numel, names));
for f = names(order)'
    v = c.(f{1});
    if ~ischar(v), v = sprintf('%.17g', v); end
    code = strrep(code, ['C_' f{1}], v);
end
end
