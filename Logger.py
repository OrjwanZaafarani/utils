class Logger():
    
    
    def __init__(self, experiment_num, print_logs=True):
        self.print_logs = print_logs
        self.experiment_num = experiment_num
        
    def log(self, thing_to_print):
        if self.print_logs == True:
            txt_file = open('runs/exp_' + str(self.experiment_num) + '/exp_' + str(self.experiment_num) + '_logs.txt', 'a+')
            txt_file = txt_file.write(str(thing_to_print))
