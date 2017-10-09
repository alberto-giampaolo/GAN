import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback


# --------------------------------------------------------------------------------------------------
class SlicePlotter(Callback):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,generator,discriminator,x_test,z_test,c_x_test=None,c_z_test=None,
                 w_x_test=None, w_z_test=None,
                 plot_every=5,
                 do_slices=False,
                 c_quantiles=[0,5,20,40,60,80,95,100], saveas=""
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.x_test = x_test
        self.z_test = z_test 
        self.c_x_test = c_x_test
        self.c_z_test = c_z_test
        self.w_x_test = w_x_test
        self.w_z_test = w_z_test
        self.do_slices = do_slices
        
        self.plot_every = plot_every
        self.has_c = type(self.c_x_test) != None
        self.saveas = saveas
        self.last_epoch = 0
        if self.has_c:
            self.c_quantiles = np.percentile(c_x_test,c_quantiles)
        
    def on_epoch_end(self, epoch, logs={}):

        self.last_epoch = epoch
        if epoch % self.plot_every != 0:
            return
        
        if self.has_c:
            x_predict = self.generator.predict([self.c_z_test,self.z_test])[1]
            x_discrim = self.discriminator.predict([self.c_x_test,self.x_test])
            z_discrim = self.discriminator.predict([self.c_z_test,x_predict])
            plot_summary_cond( self.x_test, self.c_x_test, x_predict, self.c_z_test, self.z_test ,
                               x_discrim, z_discrim,
                               target_w=self.w_x_test, generated_w=self.w_z_test,
                               c_bounds=self.c_quantiles,
                               do_slices=self.do_slices, saveas=("%sepoch_%d" %( self.saveas, epoch ) )+"_var%d.png" )
        else:
            x_predict = self.generator.predict(self.z_test)
            x_discrim = self.discriminator.predict(self.x_test)
            z_discrim = self.discriminator.predict(self.x_predict)
            if x_test.shape[-1] == 1:
                plot_summary( self.x_test.ravel(), self.x_predict.ravel(), self.z_test.ravel(), x_discrim, z_discrim) ## , solution )
            else:
                plot_summary_2d( self.x_test, x_predict, x_discrim, z_discrim )

    def on_train_end(self,logs={}):
        self.on_epoch_end(self.last_epoch)
        
    
# --------------------------------------------------------------------------------------------------
def plot_hists(target,generated,source=None,target_w=None,generated_w=None,
               bins=60,range=[-5,5],legend=True,**kwargs):
    if range is None:
        ## mins  = [target.min(),generated.min()]
        ## maxes = [target.max(),generated.max()]
        target_q = np.percentile(target,[5.,10.,90.,95.])
        generated_q = np.percentile(generated,[5.,10.,90.,95.])
        mins  = [2.*target_q[1]-target_q[0],2.*generated_q[1]-generated_q[0]]
        maxes = [2.*target_q[3]-target_q[2],2.*generated_q[3]-generated_q[2]]
        if not source is None:
            source_q = np.percentile(source,[5.,10.,90.,95.])
            mins.append(2.*source_q[1]-source_q[0])
            maxes.append(2.*source_q[3]-source_q[2])
        range = [min(mins),max(maxes)]
        print(range)
    plt.hist(generated,bins=bins,weights=generated_w,
             range=range,normed=True,label='generated',**kwargs)
    if not source is None:
        target_hist,target_edges = np.histogram(target,weights=target_w,
                                                bins=bins,range=range,normed=False)
        norm = (target_hist*(target_edges[1:]-target_edges[:-1])).sum()
        plt.bar(0.5*(target_edges[:-1]+target_edges[1:]),
                target_hist/norm,
                width=0.,
                xerr=0.5*(target_edges[1:]-target_edges[:-1]),
                yerr=np.sqrt(target_hist)/norm,
                ## ,alpha=0.5,
                label='target',**kwargs)
        plt.hist(source,weights=generated_w,
                 bins=bins,range=range,alpha=0.5,normed=True,label='source',**kwargs);
    else:
        plt.hist(target,weights=target_w,
                 bins=bins,range=range,alpha=0.5,normed=True,label='target',color='green',**kwargs);
    if legend:
        plt.legend(loc='best')
    


# --------------------------------------------------------------------------------------------------
def plot_summary(target,generated,source,target_p,generated_p,solution=None,saveas=None):

    plt.subplot(2,1,1)
    plot_hists(target,generated,range=None)
    
    plt.subplot(2,2,3)
    plot_hists(target_p,generated_p,range=[0,1])

    plt.subplot(2,2,4)
    plt.scatter(source,generated-source,label='fit')
    if solution != None:
        solution_generated,solution_source = solution
        plt.scatter(solution_source,solution_generated-solution_source,label='true',color='red')
    plt.legend(loc='best')

    if saveas != None:
        plt.savefig(saveas)
    plt.show()

# --------------------------------------------------------------------------------------------------
def plot_summary_2d(target,generated,target_p,generated_p,solution=None,saveas=None):

    plt.subplot(2,2,1)
    plt.hexbin( target[:,0,0], target[:,0,1], normed=True )
    plt.colorbar()
    
    plt.subplot(2,2,2)
    plt.hexbin( generated[:,0,0], generated[:,0,1], normed=True )
    plt.colorbar()
    
    plt.subplot(2,1,2)
    plot_hists(target_p,generated_p,range=[0,1])
    
    if saveas != None:
        plt.savefig(saveas)
    plt.show()

        
    
# --------------------------------------------------------------------------------------------------
def plot_summary_cond(target,c_target,generated,c_source,source,target_p,generated_p,
                      target_w=None,generated_w=None,
                      do_slices=False,c_bounds = [-10,-1.,-0.5,0.,0.5,1.,10.], saveas=None):
    from math import sqrt
    
    if target.shape[-1] == 1 and c_target.shape[-1] == 1 and not do_slices:
        if target.shape[0] < 100:
            n_points = int(sqrt(c_target.shape[0]))
            n_cols = 4
            n_rows = n_points // n_cols + ( n_points % n_cols != 0 )
            for ip in range(n_points):
                plt.subplot(n_rows+1,n_cols,ip+1)
                
                plt.scatter( source[ip*n_points:(ip+1)*n_points], generated[ip*n_points:(ip+1)*n_points]-source[ip*n_points:(ip+1)*n_points], label='generated', color='blue'  )
                plt.scatter( source[ip*n_points:(ip+1)*n_points], target[ip*n_points:(ip+1)*n_points]-source[ip*n_points:(ip+1)*n_points], label='target', color='red',alpha=0.5  )
            
                plt.subplot(n_rows+1,1,n_rows+1)
        else:
            allx = np.hstack([c_target.ravel(),c_source.ravel()])
            ally = np.hstack([target.ravel(),source.ravel()])
            
            xmin=np.min(allx)
            xmax=np.max(allx)
            ymin=np.min(ally)
            ymax=np.max(ally)
            
            plt.subplot(2,2,1)
            plt.hist2d( c_target.ravel(), target.ravel(), bins=60, range=[[xmin,xmax],[ymin,ymax]],
                        normed=True )
            plt.colorbar()
            
            plt.subplot(2,2,2)
            plt.hist2d( c_source.ravel(), generated.ravel(), bins=60, range=[[xmin,xmax],[ymin,ymax]],
                        normed=True )
            plt.colorbar()
            
            plt.subplot(2,1,2)
    else:
        n_rows = c_target.shape[-1]
        n_cols = len(c_bounds)-1
        for xdim in range(target.shape[-1]):
            plt.figure( figsize=(5*n_cols,5*n_rows) )
            for cdim in range(n_rows):
                for ib,bounds in enumerate(zip(c_bounds[:-1],c_bounds[1:])):
                    target_slice = ( c_target[:,:,cdim] > bounds[0] ) & ( c_target[:,:,cdim] <= bounds[1] )
                    generated_slice = ( c_source[:,:,cdim] > bounds[0] ) & ( c_source[:,:,cdim] <= bounds[1] )
                    if not target_w is None:
                        target_w_slice = target_w[target_slice.ravel()]
                    else:
                        target_w_slice = None
                    if not generated_w is None:
                        generated_w_slice = generated_w[generated_slice.ravel()]
                    else:
                        generated_w_slice = None
                    cell = ib + cdim*n_cols  + 1
                    plt.subplot( n_rows, n_cols, cell)
                    plot_hists( target[:,0,xdim][target_slice.ravel()],
                                generated[:,0,xdim][generated_slice.ravel()],
                                source[:,0,xdim][generated_slice.ravel()],
                                target_w=target_w_slice,
                                generated_w=generated_w_slice,
                                legend=False,bins=30,range=None
                    )
            if saveas != None:
                plt.savefig(saveas % xdim)
            plt.show()

        plt.figure(figsize=(5*n_cols,2.5))
    try:
        plot_hists(target_p,generated_p,target_w=target_w,generated_w=generated_w,range=None)
    except Exception:
        pass
    plt.show()

    
