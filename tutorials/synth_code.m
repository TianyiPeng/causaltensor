clear all;
diary main;

%% Get Data
load MLAB_data.txt;
data = MLAB_data;

%% Built Indices (see data description in readme file)

% California is state no 3, stored in the last column no 39
index_tr  = [39];
% 38 Control states are 1,2 & 4,5,...,38, stored in columns 1 to 38
index_co  = [1:38];

% Predcitors are stored in rows 2 to 8
index_predict = [2:8];
% Outcome Data is stored in rows 9 to 39; for 1970, 1971,...,2000
index_Y = [9:39];

%% Define Matrices for Predictors
% X0 : 7 X 38 matrix (7 smoking predictors for 38 control states)
X0 = data(index_predict,index_co);

% X1 : 10 X 1 matrix (10 crime predictors for 1 treated states)
X1 = data(index_predict,index_tr);

% Normalization (probably could be done more elegantly)
bigdata = [X0,X1];
divisor = std(bigdata');
scamatrix = (bigdata' * diag(( 1./(divisor) * eye(size(bigdata,1))) ))';
X0sca = scamatrix([1:size(X0,1)],[1:size(X0,2)]);
X1sca = scamatrix(1:size(X1,1),[size(scamatrix,2)]);
X0 = X0sca;
X1 = X1sca;
clear divisor X0sca X1sca scamatrix bigdata;

%% Define Matrices for Outcome Data
% Y0 : 31 X 38 matrix (31 years of smoking data for 38 control states)
Y0 = data(index_Y,index_co);
% Y1 : 31 X 1 matrix (31 years of smoking data for 1 treated state)
Y1 = data(index_Y,index_tr);

% Now pick Z matrices, i.e. the pretreatment period
% over which the loss function should be minmized
% Here we pick Z to go from 1970 to 1988 
 
% Z0 : 19 X 38 matrix (31 years of pre-treatment smoking data for 38 control states)
Z0 = Y0([1:19],1:38);
% Z1 : 19 X 1 matrix (31 years of pre-treatment smoking data for 1 treated state)
Z1 = Y1([1:19],1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now we implement Optimization

% Check and maybe adjust optimization settings if necessary
options = optimset('fmincon')

% Get Starting Values 
s = std([X1 X0]')';
s2 = s; s2(1)=[];
s1 = s(1);
v20 =((s1./s2).^2);

[v2,fminv,exitflag] = fmincon('loss_function',v20,[],[],[],[],...
   zeros(size(X1)),[],[],options,X1,X0,Z1,Z0);
display(sprintf('%15.4f',fminv));
v = [1;v2];
% V-weights
v

% Now recover W-weights
D = diag(v);
H = X0'*D*X0;
f = - X1'*D*X0;
options = optimset('quadprog')
[w,fval,e]=quadprog(H,f,[],[],ones(1,length(X0)),1,zeros(length(X0),1),ones(length(X0),1),[],options);
w = abs(w); 

% W-weights
w

%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Now Plot Results

Y0_plot = Y0*w;
years = [1970:2000]'; 
plot(years,Y1,'-', years,Y0_plot,'--');
axis([1970 2000 0 150]);
xlabel('year');
ylabel('smoking consumtion per capita (in packs)');
legend('Solid Real California','Dashed Synthetic California',4);

diary off;