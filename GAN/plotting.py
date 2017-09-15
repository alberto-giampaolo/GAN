import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import Callback


# --------------------------------------------------------------------------------------------------
class SlicePlotter(Callback):

    # --------------------------------------------------------------------------------------------------
    def __init__(self,generator,discriminator,x_test,z_test,c_x_test=None,c_z_test=None,plot_every=5,
                 c_quantiles=[0,5,20,40,60,80,95,100]
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.x_test = x_test
        self.z_test = z_test 
        self.c_x_test = c_x_test
        self.c_z_test = c_z_test

        self.plot_every = plot_every
        self.has_c = type(self.c_x_test) != None
        if self.has_c:
            self.c_quantiles = np.percentile(c_x_test,c_quantiles)
        
    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.plot_every != 0:
            return
        
        if self.has_c:
            x_predict = self.generator.predict([self.c_z_test,self.z_test])[1]
            x_discrim = self.discriminator.predict([self.c_x_test,self.x_test])
            z_discrim = self.discriminator.predict([self.c_z_test,x_predict])
            plot_summary_cond( self.x_test, self.c_x_test, x_predict, self.c_z_test, self.z_test , x_discrim, z_discrim, c_bounds=self.c_quantiles)
        else:
            x_predict = self.generator.predict(self.z_test)
            x_discrim = self.discriminator.predict(self.x_test)
            z_discrim = self.discriminator.predict(self.x_predict)
            if x_test.shape[-1] == 1:
                plot_summary( self.x_test.ravel(), self.x_predict.ravel(), self.z_test.ravel(), x_discrim, z_discrim) ## , solution )
            else:
                plot_summary_2d( self.x_test, x_predict, x_discrim, z_discrim )


# --------------------------------------------------------------------------------------------------
def plot_hists(target,generated,source=None,bins=60,range=[-5,5],legend=True,**kwargs):
    if range is None:
        mins  = [target.min(),generated.min()]
        maxes = [target.max(),generated.max()]
        if not source is None:
            mins.append(source.min())
            maxes.append(source.max())
        range = [min(mins),max(maxes)]
    plt.hist(generated,bins=bins,range=range,normed=True,label='generated',**kwargs)
    if not source is None:
        target_hist,target_edges = np.histogram(target,bins=bins,range=range,normed=False)
        norm = (target_hist*(target_edges[1:]-target_edges[:-1])).sum()
        plt.bar(0.5*(target_edges[:-1]+target_edges[1:]),
                target_hist/norm,
                width=0.,
                xerr=0.5*(target_edges[1:]-target_edges[:-1]),
                yerr=np.sqrt(target_hist)/norm,
                ## ,alpha=0.5,
                label='target',**kwargs)
        plt.hist(source,bins=bins,range=range,alpha=0.5,normed=True,label='source',**kwargs);
    else:
        plt.hist(target,bins=bins,range=range,alpha=0.5,normed=True,label='target',color='green',**kwargs);
    if legend:
        plt.legend(loc='best')
    


# --------------------------------------------------------------------------------------------------
def plot_summary(target,generated,source,target_p,generated_p,solution=None,saveas=None):

    plt.subplot(2,1,1)
    plot_hists(target,generated)
    
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
                      do_slices=False,c_bounds = [-10,-1.,-0.5,0.,0.5,1.,10.], saveas=None):
    ## plt.subplot(2,2,1)
    ## # plt.hexbin(c_target,target)
    ## plt.scatter(c_target,target)
    ## 
    ## plt.subplot(2,2,2)
    ## plt.scatter(c_source,generated)
    ## 
    ## plt.subplot(2,2,3)
    ## plot_hists(target_p,generated_p,range=[-1,1])
    ## 
    ## plt.subplot(2,2,4)
    ## plt.scatter(source,generated-source,label='fit')
    ## if solution != None:
    ##     solution_generated,solution_source = solution
    ##     plt.scatter(solution_source,solution_generated-solution_source,label='true',color='red')
    ## plt.legend(loc='best')
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
                    cell = ib + cdim*n_cols  + 1
                    plt.subplot( n_rows, n_cols, cell)
                    plot_hists( target[:,0,xdim][target_slice.ravel()],
                                generated[:,0,xdim][generated_slice.ravel()],
                                source[:,0,xdim][generated_slice.ravel()],
                                legend=False,bins=30
                    )
            if saveas != None:
                plt.savefig(saveas)
            plt.show()

        plt.figure(figsize=(5*n_cols,2.5))
    ## plot_hists(target_p,generated_p,range=[0,1])
    plot_hists(target_p,generated_p,range=None)
    ### if saveas != None:
    ###     plt.savefig(saveas)
    plt.show()

    
