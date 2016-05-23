%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%Creates a 6-D Matrix to store all trials, each dimension is a different 
%independent variable; the order is the same as the DataSignalVars array
%The DataSignals matrix stores each trial and its runtime as a vector
%for feature analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialize

DataMatrix = zeros(3,2,2,2,2,2);
DataSignals = [];
DataSignalVars = {'Pixels','Amount','Signals','tD','tc','tp','run_time'}';

%% Data

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 3
tp= 3

run_time= 77.725167
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 3
tp= 5

run_time= 129.511720
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10;

tD= 3
t0= 3
tc= 5
tp= 3

run_time= 128.697630
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 5
tp= 5

run_time= 213.802207
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 78.282933
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 128.942356
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 128.172487
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16];
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 214.799948
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 3
tp= 3

run_time= 129.590201
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 3
tp= 5

run_time= 214.046852
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 5
tp= 3

run_time= 213.919992
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 5
tp= 5

run_time= 354.727292
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 130.268328
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 215.433835
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 216.002077
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 360.383690
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 3
tp= 3

run_time= 260.269360
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 3
tp= 5

run_time= 430.222693
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 5
tp= 3

run_time= 427.162921
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 5
tp= 5

run_time= 710.197649
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 3
tp= 3

run_time= 261.336051
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 3
tp= 5

run_time= 431.148844
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 5
tp= 3

run_time= 429.471830
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 5
tp= 5

run_time= 710.508905
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.063318
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 129.980088
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 130.045402
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 214.299399
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 132.266036
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 216.479491
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 216.600449
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 357.687899
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);



Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 39.749756
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 66.223285
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 66.150116
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 108.966180
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 66.853490
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 109.217300
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 109.409747
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 181.433069
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 40.717586
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 66.754738
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 66.261150
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 109.167093
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 67.215118
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 110.379837
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 110.054850
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 182.398181
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.008775
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 130.605529
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 130.415262
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 216.982520
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 131.848157
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 218.003245
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 217.249081
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 360.863680
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 40.035837
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 66.642604
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 66.077687
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 109.927125
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 67.043092
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 110.416525
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 110.520779
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 182.977706
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 40.331680
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 66.374940
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);


Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 66.342109
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 109.806692
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 67.281575
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 110.601696
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 110.066731
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 182.830011
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 80.005209
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 131.752374
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 131.818146
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 217.948766
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 220.903534
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 365.298217
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 219.052569
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 134.530691
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 222.003231
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 220.622558
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 364.140275
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 156.874663
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 258.232296
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 257.146056
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 425.549121
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 261.860486
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 430.613299
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 428.570389
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 710.046425
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 40.791244
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 67.034459
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 66.862003
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 109.996798
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 68.350167
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 111.288599
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 111.450355
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 183.160959

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 3
tp= 3

run_time= 77.725167

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 3
tp= 5

run_time= 129.511720

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 5
tp= 3

run_time= 128.697630
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 3
tc= 5
tp= 5

run_time= 213.802207
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 78.282933
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 128.942356
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 128.172487
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 214.799948
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 3
tp= 3

run_time= 129.590201
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 3
tp= 5

run_time= 214.046852
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 5
tp= 3

run_time= 213.919992
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 3
tc= 5
tp= 5

run_time= 354.727292
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 130.268328
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 215.433835
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 216.002077
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 360.383690
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 3
tp= 3

run_time= 260.269360
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 3
tp= 5

run_time= 430.222693
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 5
tp= 3

run_time= 427.162921
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 3
tc= 5
tp= 5

run_time= 710.197649
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 3
tp= 3

run_time= 261.336051
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 3
tp= 5

run_time= 431.148844
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 5
tp= 3

run_time= 429.471830
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 10

tD= 10
t0= 5
tc= 5
tp= 5

run_time= 710.508905
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.063318
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 129.980088
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 130.045402
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 214.299399
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 132.266036
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 216.479491
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 216.600449
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 5
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 357.687899
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.446273
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 130.302990
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 129.747955
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 215.663589
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 132.403143
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 217.079289
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 216.633592
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 358.046457
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 155.390873
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 257.686994
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 257.074565
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 425.170932
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 258.691568
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 428.005120
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 426.344361
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);
Resolution= [16, 16]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 708.504996
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);


Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.008775
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 130.605529
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 130.415262
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 216.982520
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 131.848157
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 218.003245
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 217.249081
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 360.863680
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 79.475937
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 131.511549
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 131.283225
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 216.915181
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 133.669029
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 218.433775
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 217.753308
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [6, 6]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 362.122100
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 3

run_time= 80.623139
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 3
tp= 5

run_time= 133.098476
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 3

run_time= 132.658135
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 3
t0= 5
tc= 5
tp= 5

run_time= 218.950952
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 134.283306
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 221.588986
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 221.294189
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 25

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 365.128305
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);


Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 3

run_time= 133.050868
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 3
tp= 5

run_time= 219.399204
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 3

run_time= 219.794354
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

Resolution= [10, 10]
Classes= [0, 3, 5, 8, 9]
Amount_per_class= 10
Signals= 10

tD= 5
t0= 5
tc= 5
tp= 5

run_time= 363.736347
[DataMatrix,DataSignals] = AddData(DataMatrix,DataSignals,Resolution,Classes,Amount_per_class,...
    Signals,tD,t0,tc,tp,run_time);

