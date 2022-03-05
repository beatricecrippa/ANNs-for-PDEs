# #  Test function
# f = function(x, y, z = 1)
# {
#   coe = function(x,y) 1/2*sin(5*pi*x)*exp(-x^2)+1
#   sin(2*pi*(coe(y,1)*x*cos(z-2)-y*sin(z-2)))*cos(2*pi*(coe(y,1)*x*cos(z-2+pi/2)+coe(x,1)*y*sin((z-2)*pi/2)))
# }
# 
# locations = expand.grid(seq(0,1, length.out = 20),seq(0,1, length.out = 20))
# locations = expand.grid(runif(10,0,1),runif(10,0,1))
# mesh <-create.mesh.2D(locations)
# FEMbasis = create.FEM.basis(mesh)
# sol_exact = f(locations[,1], locations[,2])
# ran = range(sol_exact)
# data = sol_exact
 # ang = seq(5,30, length.out = 4)
area = seq(0.003,0.01, length.out = 4)
# 
# xeval=runif(10000,0,1)
# yeval=runif(10000,0,1)
# RMSE<-function(f,g) sqrt(mean((f-g)^2))
# xraff<-c(0,1,runif(18,0,1))
# yraff<-c(0,1,runif(18,0,1))
# 
# rmseNodesA=NULL
# rmseLocB=NULL
# rmseMeshB=NULL
# j=0
# for(k in seq(2.5,4,length.out = 5))
# {
#   j=j+1
#   set.seed(657489)
#   # varying number of nodes
  # xit=seq(0,1, length.out = 10^(k/2))
  # meshit <-create.mesh.2D(expand.grid(xit,xit))
  # FEMbasisit = create.FEM.basis(meshit)
#   rmse <- NULL
#   t<-NULL
#   for (i in 5:10)
#   {
#     t<-c(t,
#     microbenchmark(output_CPP<-smooth.FEM(observations=sol_exact + rnorm(length(locations), mean=0, sd=0.01*abs(ran[2]-ran[1])), locations=locations,
#                            FEMbasis=FEMbasisit, lambda.selection.criterion='newton',
#                            DOF.evaluation='exact', lambda.selection.lossfunction='GCV'),times=1)$time)
#     rmse <- c(rmse, RMSE(f(xeval,yeval),eval.FEM(output_CPP$fit.FEM,locations=cbind(xeval,yeval))))
#   }
#   rmse<-c(rmse,rmse)
#   tNodesA<-cbind(tNodesA,t)
#   rmseNodesA<-cbind(rmseNodesA,rmse)
# # 
  # #varying number of locations
  # locationsit=expand.grid(runif(10^(k/2),0,1),runif(10^(k/2),0,1))
  # sol_exactit = f(locationsit[,1], locationsit[,2])
  # ranit = range(sol_exactit)
  # tLocC <-cbind(tLocC,microbenchmark(smooth.FEM(observations=sol_exactit + rnorm(length(locationsit), mean=0, sd=0.01*abs(ranit[2]-ranit[1])), locations=locationsit,
  #                                                                    FEMbasis=FEMbasis, lambda.selection.criterion='newton',
  #                                                                    DOF.evaluation='exact', lambda.selection.lossfunction='GCV'),times=10)$time)
  # rmse <- NULL
  # for (i in 1:30)
  # {
  #   output_CPP<-smooth.FEM(observations=sol_exactit + rnorm(length(locationsit), mean=0, sd=0.01*abs(ranit[2]-ranit[1])), locations=locationsit,
  #                          FEMbasis=FEMbasis, lambda.selection.criterion='newton',
  #                          DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
  #   rmse <- c(rmse, RMSE(f(xeval,yeval),eval.FEM(output_CPP$fit.FEM,locations=cbind(xeval,yeval))))
  # }
  # rmseLocB<-cbind(rmseLocB,rmse)
for(j in 1:4)
{

  set.seed(657489)
  t<-NULL
  if(j==1)
    locraff<-expand.grid(xraff[1:6],yraff[1:7])
  # if(j==3)
  #   locraff<-expand.grid(xraff[1:7],yraff[1:9])
  # if(j==2)
  #   locraff<-expand.grid(xraff[1:14],yraff[1:15])
  if(j==4 || j==2 || j==3)
    locraff<-expand.grid(xraff[1:7],yraff[1:9])
  meshraff <-create.mesh.2D(locraff)
  meshraff<- refine.mesh.2D (meshraff, minimum_angle = 30, maximum_area=area[j])
  nodes=meshraff$nodes
  FEMbasisraff = create.FEM.basis(meshraff)
  rmse<-NULL
  for (i in 1:30)
  {
    t<-c(t,microbenchmark(output_CPP<-smooth.FEM(observations=sol_exact + rnorm(dim(locations)[1], mean=0, sd=0.01*abs(ran[2]-ran[1])),
                           FEMbasis=FEMbasisraff,locations=locations, lambda.selection.criterion='newton',
                           DOF.evaluation='exact', lambda.selection.lossfunction='GCV',
                           solver.options = "no_preconditioner"),times=1)$time)
    rmse <- c(rmse, RMSE(f(xeval,yeval),eval.FEM(output_CPP$fit.FEM,locations=cbind(xeval,yeval))))
  }
  rmseMeshA <- cbind(rmseMeshA,rmse)
  tMeshA <- cbind(tMeshA,t)#microbenchmark(smooth.FEM(observations=sol_exact + rnorm(length(locations), mean=0, sd=0.01*abs(ran[2]-ran[1])),
                        #              FEMbasis=FEMbasisraff, locations=locations, lambda.selection.criterion='newton',
                         #             DOF.evaluation='exact', lambda.selection.lossfunction='GCV'),times=30)$time)
  nnodes<-c(nnodes, dim(meshraff$nodes)[1])
}

boxplot(log(tMeshA[,4]),log(tMeshB[,4]),log(tMeshC[,4]),log(tMeshD[,4]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='log time')
 boxplot(log(tLocA[,5]),log(tLocB[,5]),log(tLocC[,4]),log(tLocD[,4]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='log time')
 boxplot(log(tLocA[,4]),log(tLocB[,4]),log(tLocC[,3]),log(tLocD[,3]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='log time')
 boxplot(log(tLocA[,5]),log(tLocB[,5]),log(tLocC[,5]), names=c('fdaPDE','mass lumping','lambda'),col=c('gray',2,3,4),ylab='log time')

 boxplot((rmseMeshA[,4]),(rmseMeshB[,4]),(rmseMeshC[,4]),(rmseMeshD[,4]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='RMSE')
 boxplot((rmseLocA[,2]),(rmseLocB[,2]),(rmseLocC[,2]),(rmseLocD[,2]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='RMSE')
 boxplot((rmseLocA[,3]),(rmseLocB[,3]),(rmseLocC[,3]),(rmseLocD[,3]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='RMSE')
 boxplot((rmseLocA[,4]),(rmseLocB[,4]),(rmseLocC[,4]),(rmseLocD[,4]), names=c('fdaPDE','mass lumping','lambda','block'),col=c('gray',2,3,4),ylab='RMSE')
 
 
plot(area,colMeans(rmseMeshA),col=1,pch=19,xlab = 'maximum elements area', ylab='RMSE',ylim=c(min(colMeans(rmseMeshB)),max(colMeans(rmseMeshA))))
lines(area,colMeans(rmseMeshA),col=1,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,colMeans(rmseMeshB),col=2,pch=4,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(area,colMeans(rmseMeshB),col=2,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,colMeans(rmseMeshC),col=3,pch=1,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(seq(2.5,3.625,length.out = 4),colMeans(rmseMeshC),col=3,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,colMeans(rmseMeshD),col=4,pch=4,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(seq(2.5,3.625,length.out = 4),colMeans(rmseMeshD),col=4,xlab = 'minimum mesh angle', ylab='mean RMSE')
grid()
legend('topleft',legend=c('no preconditioner', 'mass lumping',' lambda preconditioner', 'block preconditioner'), cex=0.62,col=1:4, pch=19)

plot(area,log(colMeans(tMeshA)),col=1,pch=19,xlab = 'maximum elements area', ylab='log time',ylim=c(min(colMeans(log(tMeshA))),max(colMeans(log(tMeshD))+0.15)))
lines(area,log(colMeans(tMeshA)),col=1,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,log(colMeans(tMeshB)),col=2,pch=19,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(area,log(colMeans(tMeshB)),col=2,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,log(colMeans(tMeshC)),col=3,pch=19,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(area,log(colMeans(tMeshC)),col=3,xlab = 'minimum mesh angle', ylab='mean RMSE')
points(area,log(colMeans(tMeshD)),col=4,pch=19,xlab = 'minimum mesh angle', ylab='mean RMSE')
lines(area,log(colMeans(tMeshD)),col=4,xlab = 'minimum mesh angle', ylab='mean RMSE')
grid()
legend('topleft',legend=c('no preconditioner', 'mass lumping',' lambda preconditioner', 'block preconditioner'), cex=0.62,col=1:4, pch=19)

