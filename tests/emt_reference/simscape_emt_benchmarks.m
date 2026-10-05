function simscape_emt_benchmarks()
% SIMSCAPE_EMT_BENCHMARKS  Reference waveforms for the EMT study of DC networks.
%
% Builds each case in emt_benchmark_params.json as a Simscape (Foundation
% Library) model - the file tests/test_emt.py builds Electrisim's circuits
% from - simulates it, and writes its waveforms to emt_<case>.csv. Run it
% again when the solver or a benchmark changes; needs Simscape.
%
% Cases:
%   long_cable          a 2 km cable as 10 pi sections, energised, then faulted at its far end
%   cpl_95, cpl_105     a constant-power load at 95 % and 105 % of its stability limit
%   breaker_solid_state, breaker_mechanical
%                       a breaker opening into its surge arrester, early and late
%
% Switches and arresters are as Electrisim's: a switch is R_on closed and
% R_off open; an arrester is R_off until its clamping voltage, then that
% voltage behind R_on (each way: a diode, the clamping voltage and R_on, in
% parallel with R_off). Simscape's diode needs a forward voltage above
% zero: 1 uV.

here = fileparts(mfilename('fullpath'));
p = jsondecode(fileread(fullfile(here, 'emt_benchmark_params.json')));
long_cable(p, here);
for frac = p.cpl.fractions'
    cpl(p, frac, here);
end
for name = fieldnames(p.breaker.cases)'
    breaker(p, name{1}, here);
end
end

% --- the cases ------------------------------------------------------------------

function long_cable(p, here)
q = p.long_cable;
m = Model('emt_long_cable');
src = m.node(); a = m.node();
m.two(m.dcsrc(q.E), src, m.gnd);
r_s = m.res(q.Rs); s_mid = m.node();
m.two(r_s, src, s_mid);
m.two(m.ind(q.Ls, 0), s_mid, a);
n = q.sections; r = q.r_per_km * q.km / n; l = q.l_per_km * q.km / n; c = q.c_per_km * q.km / n;
nodes = {a};
for k = 1:n
    nodes{k + 1} = m.node();
end
caps = cell(1, n + 1);
for k = 1:n + 1
    share = 1; if k == 1 || k == n + 1, share = 0.5; end
    caps{k} = m.cap(share * c, 0);
    m.two(caps{k}, nodes{k}, m.gnd);
end
for k = 1:n
    mid = m.node();
    m.two(m.res(r), nodes{k}, mid);
    m.two(m.ind(l, 0), mid, nodes{k + 1});
end
b = nodes{n + 1};
m.two(m.res(q.R_load), b, m.gnd);
f = m.node();
sw = m.switch_at(q.t_fault, true, p);
m.two(sw, b, f);
r_f = m.res(q.Rf);
m.two(r_f, f, m.gnd);
log = m.run(q.t_end, 1e-7);
t = unique([0:1e-6:q.t_end])';
write(here, 'emt_long_cable.csv', t, {'i_source', 'v_end', 'i_fault'}, ...
    {m.series(log, r_s, 'i', t), m.series(log, caps{end}, 'v', t), m.series(log, r_f, 'i', t)});
m.done();
end

function cpl(p, frac, here)
q = p.cpl;
v0 = q.E / (1 + frac * q.R ^ 2 * q.C / q.L);
P = v0 * (q.E - v0) / q.R;
m = Model(sprintf('emt_cpl_%d', round(100 * frac)));
src = m.node(); mid = m.node(); n = m.node();
m.two(m.dcsrc(q.E), src, m.gnd);
m.two(m.res(q.R), src, mid);
m.two(m.ind(q.L, P / v0), mid, n);
c = m.cap(q.C, v0);
m.two(c, n, m.gnd);
m.cpl_load(n, P, P * (1 + q.step), q.t_step);
log = m.run(q.t_end, 1e-5);
t = unique([0:1e-5:q.t_end])';
write(here, sprintf('emt_cpl_%d.csv', round(100 * frac)), t, {'v_load'}, {m.series(log, c, 'v', t)});
m.done();
end

function breaker(p, name, here)
q = p.breaker; c = q.cases.(name);
m = Model(['emt_breaker_' name]);
src = m.node(); s_mid = m.node(); a = m.node(); contacts = m.node(); term = m.node(); c_mid = m.node(); b = m.node();
m.two(m.dcsrc(q.E), src, m.gnd);
m.two(m.res(q.Rs), src, s_mid);
m.two(m.ind(q.Ls, 0), s_mid, a);
l_lim = m.ind(q.L_lim, 0);
m.two(l_lim, a, contacts);
sw = m.switch_at(c.t_open, false, p);
m.two(sw, contacts, term);
m.arrester(contacts, term, q.Vc, p);
m.two(m.res(q.R_cable), term, c_mid);
m.two(m.ind(q.L_cable, 0), c_mid, b);
m.two(m.res(q.R_load), b, m.gnd);
f = m.node();
m.two(m.switch_at(q.t_fault, true, p), b, f);
r_f = m.res(q.Rf);
m.two(r_f, f, m.gnd);
log = m.run(c.t_end, 1e-7);
t = unique([0:1e-6:c.t_end])';
write(here, ['emt_breaker_' name '.csv'], t, {'i_breaker', 'v_contacts', 'i_fault'}, ...
    {m.series(log, l_lim, 'i', t), m.series(log, sw, 'v', t), m.series(log, r_f, 'i', t)});
m.done();
end

function write(here, file, t, names, cols)
tbl = array2table([t, cols{:}], 'VariableNames', [{'t'}, names]);
writetable(tbl, fullfile(here, file));
fprintf('Wrote %s: %d rows\n', file, height(tbl));
end

% --- building a Simscape model ----------------------------------------------------

function m = Model(name)
fl = 'fl_lib/Electrical/Electrical Elements/';
src = 'fl_lib/Electrical/Electrical Sources/';
if bdIsLoaded(name), close_system(name, 0); end
new_system(name);
count = 0;
nodes = {};
m.gnd = [];
m.node = @node;
m.two = @two;
m.res = @res;
m.ind = @ind;
m.cap = @cap;
m.dcsrc = @dcsrc;
m.switch_at = @switch_at;
m.arrester = @arrester;
m.cpl_load = @cpl_load;
m.run = @run;
m.series = @series;
m.done = @done;
ref = place([fl 'Electrical Reference']);
m.gnd = node();
connect(m.gnd, ports(ref).LConn(1));
solver = place('nesl_utility/Solver Configuration');
connect(m.gnd, ports(solver).RConn(1));

    function h = place(lib)
        count = count + 1;
        h = add_block(lib, sprintf('%s/b%d', name, count));
    end
    function pp = ports(h)
        pp = get_param(h, 'PortHandles');
    end
    function k = node()
        nodes{end + 1} = [];
        k = numel(nodes);
    end
    function connect(k, port)
        if isempty(nodes{k})
            nodes{k} = port;
        else
            add_line(name, nodes{k}, port, 'autorouting', 'off');
        end
    end
    function two(h, a, b)
        % Block h's first electrical port on node a, its second on node b.
        pp = ports(h);
        e = electrical(h, pp);
        connect(a, e(1));
        connect(b, e(2));
    end
    function e = electrical(h, pp)
        % The switch, sensor and controlled source have a physical-signal port
        % before their - terminal on the bottom: RConn(2) is the electrical one.
        if any(strcmp(get_param(h, 'ReferenceBlock'), {[fl 'Switch']}))
            e = [pp.LConn(1), pp.RConn(2)];
        else
            e = [pp.LConn(1), pp.RConn(1)];
        end
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
    function h = dcsrc(v)
        h = place([src 'DC Voltage Source']);
        set_param(h, 'v0', num2str(v, 17));
    end
    function h = switch_at(t, closed_after, p)
        % A switch that closes (closed_after) or opens at t.
        h = place([fl 'Switch']);
        set_param(h, 'R_closed', num2str(p.switch_r_on, 17), 'G_open', num2str(1 / p.switch_r_off, 17), 'Threshold', '0.5');
        step = place('simulink/Sources/Step');
        set_param(step, 'Time', num2str(t, 17), 'Before', num2str(~closed_after), 'After', num2str(closed_after));
        conv = place('nesl_utility/Simulink-PS Converter');
        add_line(name, ports(step).Outport(1), ports(conv).Inport(1), 'autorouting', 'off');
        pp = ports(h);
        add_line(name, ports(conv).RConn(1), pp.RConn(1), 'autorouting', 'off');
    end
    function arrester(a, b, vc, p)
        % R_off across it, and each way a diode, the clamping voltage and R_on.
        two(res(p.arrester_r_off), a, b);
        for dir = [1, -1]
            d = place([fl 'Diode']);
            set_param(d, 'Vf', '1e-6', 'Ron', '1e-9', 'Goff', '1e-12');
            x = node(); y = node();
            if dir > 0
                two(d, a, x); two(dcsrc(vc), x, y); two(res(p.arrester_r_on), y, b);
            else
                two(d, b, x); two(dcsrc(vc), x, y); two(res(p.arrester_r_on), y, a);
            end
        end
    end
    function cpl_load(n, p0, p1, t_step)
        % i = P(t) / v from node n to the reference: P steps from p0 to p1 at t_step.
        sense = place('fl_lib/Electrical/Electrical Sensors/Voltage Sensor');
        sp = ports(sense);
        connect(n, sp.LConn(1));
        connect(m.gnd, sp.RConn(2));
        step = place('simulink/Sources/Step');
        set_param(step, 'Time', num2str(t_step, 17), 'Before', num2str(p0, 17), 'After', num2str(p1, 17));
        conv = place('nesl_utility/Simulink-PS Converter');
        set_param(conv, 'Unit', 'W');
        add_line(name, ports(step).Outport(1), ports(conv).Inport(1), 'autorouting', 'off');
        div = place('fl_lib/Physical Signals/Functions/PS Divide');
        dp = ports(div);
        add_line(name, ports(conv).RConn(1), dp.LConn(1), 'autorouting', 'off');
        add_line(name, sp.RConn(1), dp.LConn(2), 'autorouting', 'off');
        isrc = place([src 'Controlled Current Source']);
        ip = ports(isrc);
        add_line(name, dp.RConn(1), ip.RConn(1), 'autorouting', 'off');
        % The current flows from its tail (bottom) to its head (top): the node on the tail.
        connect(n, ip.RConn(2));
        connect(m.gnd, ip.LConn(1));
    end
    function log = run(t_end, max_step)
        set_param(name, 'SimscapeLogType', 'all', 'SimscapeLogName', 'simlog', 'SimscapeLogLimitData', 'off', ...
            'StopTime', num2str(t_end, 17), 'Solver', 'ode23t', 'MaxStep', num2str(max_step, 17), ...
            'RelTol', '1e-7', 'AbsTol', '1e-6');
        out = sim(name, 'ReturnWorkspaceOutputs', 'on');
        log = out.get('simlog');
    end
    function done()
        close_system(name, 0);
    end
    function y = series(log, h, var, t)
        s = log.(get_param(h, 'Name')).(var).series;
        [tt, k] = unique(s.time, 'last');
        vals = s.values;
        y = interp1(tt, vals(k), t);
    end
end
