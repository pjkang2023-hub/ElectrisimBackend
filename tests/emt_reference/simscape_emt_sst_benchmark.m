function simscape_emt_sst_benchmark(which)
% SIMSCAPE_EMT_SST_BENCHMARK  Reference waveforms for an SST through a fault on its LV DC port.
%
% Builds the circuit in emt_sst_benchmark_params.json in Simscape (Foundation
% Library): a 10 kV grid behind R_g and L_g (R_p across L_g), filter
% capacitors at the rectifier's AC bus; the rectifier's averaged phase
% voltages behind its reactor, its DC link capacitor and the current its AC
% side delivers into it (-p / v_dc), behind r_dc to the link bus; the DC/DC
% stage - a dual active bridge, averaged - drawing its input current from its
% input capacitor (behind r_in to the link bus) and delivering its output
% current into its output capacitor (behind r_out to the LV DC bus); a
% resistive load, and a fault through r_fault switched on (and off). The
% controllers - the rectifier's (PLL, dq current control, DC voltage loop, Q
% at zero, current limit) and the bridge's (output voltage loop, its output
% current fed forward, current limit, its phase shift and the average
% currents it gives, blocking on undervoltage) - run in a MATLAB Function
% block every dt, as Electrisim's. Both start in the circuit's steady state,
% which this script computes as tests/test_emt_sst.py does. Writes
% emt_sst_<case>.csv. Needs Simscape.
%
% Cases (all by default, or the one named): 'limited', 'bolted'.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_sst_benchmark_params.json')));
if nargin < 1, which = fieldnames(p.cases)'; end
if ischar(which), which = {which}; end
for k = 1:numel(which)
    run_case(p, which{k}, here);
end
end

function run_case(p, name, here)
c = p.cases.(name);
s = steady(p);
w = 2 * pi * p.f_hz;
g = p.grid; rc = p.rectifier; d = p.dab;
mdl = ['emt_sst_' name];
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

% The controllers: their inputs sampled every dt, their outputs held over the next.
ctrl = place('simulink/User-Defined Functions/MATLAB Function');
set_param(ctrl, 'Name', 'ctrl');
ctrl = [mdl '/ctrl'];
chart = find(sfroot, '-isa', 'Stateflow.EMChart', 'Path', ctrl);
chart.Script = controller_code(p, s);
mux = place('simulink/Signal Routing/Mux');
set_param(mux, 'Inputs', '11');
zoh = place('simulink/Discrete/Zero-Order Hold');
set_param(zoh, 'SampleTime', num2str(p.dt, 17));
% Their measurements 1 ns late: Simscape takes its network's outputs to depend
% on its inputs at once - an algebraic loop through the controllers - though
% each signal measured is a state, the same 1 ns before.
lag = place('simulink/Continuous/Transport Delay');
set_param(lag, 'DelayTime', '1e-9', 'InitialOutput', '0');
add_line(mdl, P(mux).Outport(1), P(lag).Inport(1));
add_line(mdl, P(lag).Outport(1), P(zoh).Inport(1));
add_line(mdl, P(zoh).Outport(1), P(ctrl).Inport(1));
dmx = place('simulink/Signal Routing/Demux');
set_param(dmx, 'Outputs', '6');
add_line(mdl, P(ctrl).Outport(1), P(dmx).Inport(1));

ang = -(0:2) * 2 * pi / 3;
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
    c_f(k) = cap(g.c_f, real(s.V * exp(1i * ang(k))));
    two(c_f(k), pcc, gnd);
    two_sensor(sensor([sens 'Voltage Sensor'], 'V', mux, k), pcc, gnd);
    % The rectifier's phase: its averaged voltage, its reactor, its current into its AC bus.
    controlled(place([srcs 'Controlled Voltage Source']), P(dmx).Outport(k), 'V', gnd, x);
    y = node(); z = node();
    two(res(rc.r), x, y);
    l_c(k) = ind(rc.l, real(s.Ic * exp(1i * ang(k))));
    two(l_c(k), y, z);
    two_sensor(sensor([sens 'Current Sensor'], 'A', mux, 3 + k), z, pcc);
end

% The rectifier's DC side: its link capacitor, the current its AC side delivers, r_dc to the link bus.
dcp = node(); m1 = node(); link = node();
c_link = cap(rc.c_link, rc.v_dc);
two(c_link, dcp, gnd);
controlled(place([srcs 'Controlled Current Source']), P(dmx).Outport(4), 'A', gnd, dcp);
two_sensor(sensor([sens 'Voltage Sensor'], 'V', mux, 7), dcp, gnd);
two_sensor(sensor([sens 'Current Sensor'], 'A', mux, 8), dcp, m1);
two(res(rc.r_dc), m1, link);
% The bridge: its input capacitor behind r_in, its output capacitor behind r_out, the currents it draws and delivers.
din = node(); dout = node(); m2 = node(); lv = node();
two(res(d.r_in), link, din);
c_in = cap(s.c_in, s.v_link);
two(c_in, din, gnd);
controlled(place([srcs 'Controlled Current Source']), P(dmx).Outport(5), 'A', din, gnd);
two_sensor(sensor([sens 'Voltage Sensor'], 'V', mux, 9), din, gnd);
c_out = cap(s.c_out, d.v_out);
two(c_out, dout, gnd);
controlled(place([srcs 'Controlled Current Source']), P(dmx).Outport(6), 'A', gnd, dout);
two_sensor(sensor([sens 'Voltage Sensor'], 'V', mux, 10), dout, gnd);
i_dab = sensor([sens 'Current Sensor'], 'A', mux, 11);
two_sensor(i_dab, dout, m2);
two(res(d.r_out), m2, lv);
% The LV DC bus: its load, and the fault.
two(res(p.load.r_load), lv, gnd);
sw = place([fl 'Switch']);
set_param(sw, 'R_closed', num2str(p.switch_r_on, 17), 'G_open', num2str(1 / p.switch_r_off, 17), 'Threshold', '0.5');
on = place('simulink/Sources/Step');
set_param(on, 'Time', num2str(c.t_on, 17), 'Before', '0', 'After', '1');
sig = P(on).Outport(1);
if ~isempty(c.t_off)
    off = place('simulink/Sources/Step');
    set_param(off, 'Time', num2str(c.t_off, 17), 'Before', '0', 'After', '1');
    sub = place('simulink/Math Operations/Sum');
    set_param(sub, 'Inputs', '+-');
    add_line(mdl, sig, P(sub).Inport(1));
    add_line(mdl, P(off).Outport(1), P(sub).Inport(2));
    sig = P(sub).Outport(1);
end
cv = place('nesl_utility/Simulink-PS Converter');
add_line(mdl, sig, P(cv).Inport(1));
add_line(mdl, P(cv).RConn(1), P(sw).RConn(1));
f = node();
pp = P(sw);
connect(lv, pp.LConn(1));
connect(f, pp.RConn(2));
r_f = res(c.r_fault);
two(r_f, f, gnd);
lv_v = place([sens 'Voltage Sensor']);
two_sensor(lv_v, lv, gnd);

set_param(mdl, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
    'StopTime', num2str(c.t_end, 17), 'Solver', 'ode23t', 'MaxStep', num2str(p.dt, 17), 'RelTol', '1e-7', 'AbsTol', '1e-6');
out = sim(mdl, 'ReturnWorkspaceOutputs', 'on', 'TimeOut', 900);
log = out.get('simlog');
t = (0:p.dt:c.t_end)';
pw = zeros(size(t));
for k = 1:3
    pw = pw + series(log, c_f(k), 'v', t) .* series(log, l_c(k), 'i', t);
end
cols = [t, series(log, c_out, 'v', t), series(log, c_link, 'v', t), series(log, i_dab, 'I', t), ...
        series(log, r_f, 'i', t), pw];
tbl = array2table(cols, 'VariableNames', {'t', 'v_out', 'v_link', 'i_dab', 'i_fault', 'p_ac'});
writetable(tbl, fullfile(here, ['emt_sst_' name '.csv']));
fprintf('Wrote emt_sst_%s.csv: %d rows\n', name, height(tbl));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', mdl, count));
    end
    function k_ = node()
        nodes{end + 1} = [];
        k_ = numel(nodes);
    end
    function connect(k_, port)
        if isempty(nodes{k_})
            nodes{k_} = port;
        else
            add_line(mdl, nodes{k_}, port, 'autorouting', 'off');
        end
    end
    function two(h, a_, b_)
        pp_ = P(h);
        connect(a_, pp_.LConn(1));
        connect(b_, pp_.RConn(1));
    end
    function two_sensor(h, a_, b_)
        % A sensor's + on a_, its - on b_; its measurement is RConn(1).
        pp_ = P(h);
        connect(a_, pp_.LConn(1));
        connect(b_, pp_.RConn(2));
    end
    function h = sensor(lib, unit, mx, port)
        h = place(lib);
        cv_ = place('nesl_utility/PS-Simulink Converter');
        set_param(cv_, 'Unit', unit);
        add_line(mdl, P(h).RConn(1), P(cv_).LConn(1));
        add_line(mdl, P(cv_).Outport(1), P(mx).Inport(port));
    end
    function controlled(h, from, unit, a_, b_)
        % A controlled source driven by a Simulink signal: its + (or its
        % head, for a current source) on b_, its - (tail) on a_.
        cv_ = place('nesl_utility/Simulink-PS Converter');
        set_param(cv_, 'Unit', unit);
        add_line(mdl, from, P(cv_).Inport(1));
        pp_ = P(h);
        add_line(mdl, P(cv_).RConn(1), pp_.RConn(1));
        connect(b_, pp_.LConn(1));
        connect(a_, pp_.RConn(2));
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
    function h = cap(cc, v0)
        h = place([fl 'Capacitor']);
        set_param(h, 'c', num2str(cc, 17), 'r', '0', 'g', '0', ...
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
% Its steady state, as the test computes it: the LV load's current from the
% bridge's output capacitor at v_out; the bridge's input power; the link
% bus's voltage behind r_dc; then the rectifier's AC bus voltage V and
% current Ic (in phase with V: Q = 0) by fixed point.
g = p.grid; rc = p.rectifier; d = p.dab;
w = 2 * pi * p.f_hz;
s.i_o = d.v_out / (p.load.r_load + d.r_out);
s.p_out = d.v_out * s.i_o;
s.p_in = s.p_out / d.eta;
v_bus = rc.v_dc;
for it = 1:50
    i_dc = s.p_in / v_bus;
    v_bus = rc.v_dc - rc.r_dc * i_dc;
end
s.i_dc = i_dc; s.v_link = v_bus;
z = g.v_ll ^ 2 / g.s_sc;
s.r_g = z / sqrt(1 + g.xr ^ 2);
x_g = g.xr * s.r_g;
s.l_g = x_g / w;
zg = s.r_g + 1i * x_g * g.r_p / (g.r_p + 1i * x_g);
s.E = sqrt(2 / 3) * g.v_ll;
p_dc = rc.v_dc * i_dc;
V = s.E;
for it = 1:200
    a = abs(V);
    x = (-a + sqrt(a * a - 4 * rc.r * p_dc / 1.5)) / (2 * rc.r);
    Ic = x * V / a;
    V = s.E - zg * (1i * w * g.c_f * V - Ic);
end
s.V = V; s.Ic = Ic;
s.Ig = (s.E - V) / zg;
s.Ec = V + (rc.r + 1i * w * rc.l) * Ic;
% The bridge's capacitors: 2 ms of its rating stored at each side's nominal voltage.
s.c_in = 4e-3 * d.p_rated / rc.v_dc ^ 2;
s.c_out = 4e-3 * d.p_rated / d.v_out ^ 2;
end

function code = controller_code(p, s)
% The MATLAB Function block's code: Electrisim's Vsc.control (the rectifier)
% and DcDc.control (the bridge, averaged), with their constants and starting
% states written in.
rc = p.rectifier; d = p.dab;
w0 = 2 * pi * p.f_hz;
v_ll = p.grid.v_ll;
c = struct();
c.w0 = w0; c.dt = p.dt; c.r = rc.r; c.l = rc.l;
c.i_max = rc.i_limit_pu * sqrt(2) * rc.s_rated / (sqrt(3) * v_ll);
c.v_nom_peak = sqrt(2 / 3) * v_ll;
th = angle(s.V);
v_pk = abs(s.V);
c.theta0 = th;
c.kp_pll = 2 * 0.7 * (2 * pi * 20) / v_pk; c.ki_pll = (2 * pi * 20) ^ 2 / v_pk;
a_c = 2 * pi * 500;
c.kp_i = a_c * rc.l; c.ki_i = c.kp_i * a_c / 10;
i_d = abs(s.Ic) * cos(angle(s.Ic) - th); i_q = abs(s.Ic) * sin(angle(s.Ic) - th);
e_d = abs(s.Ec) * cos(angle(s.Ec) - th); e_q = abs(s.Ec) * sin(angle(s.Ec) - th);
c.int_d0 = e_d - (v_pk + rc.r * i_d - w0 * rc.l * i_q);
c.int_q0 = e_q - (rc.r * i_q + w0 * rc.l * i_d);
w_v = 2 * pi * 30;
c.kp_v = 2 * 0.7 * w_v * rc.c_link; c.ki_v = w_v * w_v * rc.c_link;
c.v_dc_ref = rc.v_dc;
c.q_out_ref = -1.5 * v_pk * i_q;
c.int_v0 = -1.5 * v_pk * i_d / rc.v_dc - s.i_dc;
e0 = real(s.Ec * exp(-1i * (0:2) * 2 * pi / 3));
% The bridge.
c.n = rc.v_dc / d.v_out;
c.f_sw = d.f_sw;
c.l_lk = d.v_out ^ 2 * 5 / (72 * d.f_sw * d.p_rated);
c.i_max_d = d.i_limit_pu * d.p_rated / d.v_out;
w_d = 2 * pi * 300;
c.kp_d = 2 * 0.7 * w_d * s.c_out; c.ki_d = w_d * w_d * s.c_out;
c.v_ref_d = d.v_out;
c.block_in = d.block_pu * rc.v_dc; c.block_out = d.block_pu * d.v_out;
c.eta = d.eta;
c.y0 = sprintf('[%.17g; %.17g; %.17g; %.17g; %.17g; %.17g]', [e0, s.i_dc, s.p_in / s.v_link, s.p_out / d.v_out]);
lines = {
'function y = ctrl(u)'
'% u: AC bus voltages (3), rectifier currents into it (3), its v_dc, its DC current out,'
'%    the bridge''s input voltage, its output voltage, its output current to the LV bus.'
'% y: the rectifier''s phase voltages (3), its DC current; the bridge''s input and output currents.'
'persistent th w int_pll int_d int_q int_v int_b blocked started'
'if isempty(started)'
'    th = C_theta0; w = C_w0; int_pll = 0; int_d = C_int_d0; int_q = C_int_q0; int_v = C_int_v0;'
'    int_b = 0; blocked = false; started = false;'
'end'
'if ~started'
'    % Its first sample, at t = 0: the steady state''s outputs.'
'    started = true;'
'    y = C_y0;'
'    return'
'end'
'dt = C_dt;'
'% --- the rectifier ---'
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
'i_rect = -sum(e .* i) / max(v_dc, 1);'
'% --- the bridge ---'
'v_in = u(9); v_out = u(10); i_out = u(11);'
'if ~blocked && (v_in < C_block_in || v_out < C_block_out), blocked = true; end'
'if blocked'
'    y = [e; i_rect; 0; 0];'
'    return'
'end'
'err_b = C_v_ref_d - v_out;'
'i_ref = i_out + C_kp_d * err_b + int_b;'
'lim_b = i_ref < -C_i_max_d || i_ref > C_i_max_d;'
'i_ref = min(max(i_ref, -C_i_max_d), C_i_max_d);'
'v1 = max(v_in, 1) / C_n;'
'x = 2 * pi ^ 2 * C_f_sw * C_l_lk * abs(i_ref) / v1;'
'if x >= pi ^ 2 / 4'
'    phi = pi / 2; lim_b = true;'
'else'
'    phi = 0.5 * (pi - sqrt(pi ^ 2 - 4 * x));'
'end'
'if i_ref < 0, phi = -phi; end'
'if ~lim_b, int_b = int_b + C_ki_d * err_b * dt; end'
'i_o = max(v_in, 0) / C_n * phi * (pi - abs(phi)) / (2 * pi ^ 2 * C_f_sw * C_l_lk);'
'p_o = v_out * i_o;'
'if p_o >= 0, p_i = p_o / C_eta; else, p_i = p_o * C_eta; end'
'y = [e; i_rect; p_i / max(v_in, 1); i_o];'
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
