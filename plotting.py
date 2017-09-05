import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------------------------------
def plot_hists(target,generated,source=None,bins=60,range=[-5,5],legend=True,**kwargs):
    plt.hist(generated,bins=bins,range=range,normed=True,label='generated',**kwargs)
    plt.hist(target,bins=bins,range=range,alpha=0.5,normed=True,label='target',**kwargs);
    if type(source) != type(None):
        plt.hist(source,bins=bins,range=range,alpha=0.5,normed=True,label='source',**kwargs);
    if legend:
        plt.legend(loc='best')
    


# ------------------------------------------------------------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------------------------------------------------------------
def plot_summary_2d(target,generated,target_p,generated_p,solution=None,saveas=None):

    plt.subplot(2,2,1)
    plt.hexbin( target[:,0,0], target[:,0,1] )
    
    plt.subplot(2,2,2)
    plt.hexbin( generated[:,0,0], generated[:,0,1] )
    
    plt.subplot(2,1,2)
    plot_hists(target_p,generated_p,range=[0,1])
    
    if saveas != None:
        plt.savefig(saveas)
    plt.show()

        
    
# ------------------------------------------------------------------------------------------------------------------------------------------
def plot_summary_cond(target,c_target,generated,c_source,source,target_p,generated_p,
                      do_slices=False,c_bounds = [-10,-1.,-0.5,0.,0.5,1.,10.]):
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
            plt.subplot(2,2,1)
            plt.hexbin( c_target.ravel(), target.ravel() )
            
            plt.subplot(2,2,2)
            plt.hexbin( c_source.ravel(), generated.ravel() )
            
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
            plt.show()

        plt.figure(figsize=(5*n_cols,2.5))
    plot_hists(target_p,generated_p,range=[0,1])
    ## if saveas != None:
    ##     plt.savefig(saveas)
    plt.show()

    
