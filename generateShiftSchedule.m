%% plot the shift schedule
shiftInitSpd = [3, 7, 12, 15, 22];
shiftEndSpd = [7, 11, 18, 23, 27];
figure
hold on
aaa = (-3:0.01:3)';
for i = 1:5
    plot(aaa, max(min(2*i*aaa+shiftInitSpd(i),shiftEndSpd(i)),shiftInitSpd(i)))
end
grid on
xlabel('acceleration (m/s^2)')
ylabel('speed (m/s)')
xticks([-3:0.1:3]);
yticks([0:30]);

%% Create a look up table
vvv = (0:0.1:30)';
[vGrid, aGrid] = meshgrid(vvv, aaa); % one line after another
gGrid = ones(size(vGrid));
for i = 1:numel(aaa)
    pts = max(min(2*(1:numel(shiftInitSpd))*aGrid(i,1)+shiftInitSpd,shiftEndSpd),shiftInitSpd);
    idx = [1,max(min(round(pts/0.1+1),numel(vvv)),1),numel(vvv)];
    for j = 1:(numel(idx)-1)
        gGrid(i,idx(j):idx(j+1))=j;
    end
end

figure;
mesh(vGrid, aGrid, gGrid)

% you can then use interp2 to find gear position, given any speed and
% acceleration
ShiftSchedule.Speed = vGrid;
ShiftSchedule.Acceleration = aGrid;
ShiftSchedule.Gear = gGrid;
ShiftSchedule.description = 'round(interp2(ShiftSchedule.Speed, ShiftSchedule.Acceleration, ShiftSchedule.Gear, v, a)); v (m/s), a (m/s^2)';

