import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------------------------------------------------------------------
def plot_hists(target,generated,bins=60,range=[-5,5],**kwargs):
    plt.hist(generated,bins=bins,range=range,normed=True,label='generated',**kwargs)
    plt.hist(target,bins=bins,range=range,alpha=0.5,normed=True,label='target',**kwargs);
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
def plot_summary_cond(target,c_target,generated,c_source,source,target_p,generated_p):##,solution=None,saveas=None):

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
    n_points = int(sqrt(c_target.shape[0]))
    n_cols = 4
    n_rows = n_points // n_cols + ( n_points % n_cols != 0 )
    for ip in range(n_points):
        plt.subplot(n_rows+1,n_cols,ip+1)
        
        plt.scatter( source[ip*n_points:(ip+1)*n_points], generated[ip*n_points:(ip+1)*n_points]-source[ip*n_points:(ip+1)*n_points], label='generated', color='blue'  )
        plt.scatter( source[ip*n_points:(ip+1)*n_points], target[ip*n_points:(ip+1)*n_points]-source[ip*n_points:(ip+1)*n_points], label='target', color='red',alpha=0.5  )
        
        ## plt.scatter( c_source[ip*n_points:(ip+1)*n_points], generated[ip*n_points:(ip+1)*n_points], label='generated', color='blue'  )
        ## plt.scatter( c_target[ip*n_points:(ip+1)*n_points], target[ip*n_points:(ip+1)*n_points], label='target', color='red'  )
        # plt.legend(loc='best')
    
    plt.subplot(n_rows+1,1,n_rows+1)
    plot_hists(target_p,generated_p,range=[-1,1])
    ## if saveas != None:
    ##     plt.savefig(saveas)
    plt.show()

        
    
