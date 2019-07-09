import os
def mkdir ():
    plot_dir = "plots"
    model_dir = "model"
    data_dir = "data"
    log_dir = "log"
    if not os.path.exists(plot_dir):
        print ("Creating directory for {} ".format(plot_dir))
        os.makedirs(plot_dir)
    if not os.path.exists(model_dir):
        print("Creating directory for {} ".format(model_dir))
        os.makedirs(model_dir)
    if not os.path.exists(data_dir):
        print("Creating directory for {} ".format(data_dir))
        os.makedirs(data_dir)
    if not os.path.exists(log_dir):
        print("Creating directory for {} ".format(log_dir))
        os.makedirs(log_dir)
    return (print ("Dirs Checked Sucessfully!"))
