function ssr = sumsq(v2,X1,X0,Z1,Z0)
  v = [1;v2];
  D = diag(v);
  H = X0'*D*X0;
  f = - X1'*D*X0;
  l = size(Z0,2);
  [w,fval,e]=quadprog(H,f,[],[],ones(1,l),1,zeros(l,1),ones(l,1));
  w = abs(w);
  e = Z1 - Z0*w;
  ssr = sum(e.^2);
